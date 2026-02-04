import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 独立函数部分 ========= #
def dice_loss(pred, target):
    """二分类 Dice loss；pred/target: [B, H, W] 已展平"""
    smooth = 1.0
    size = pred.size(0)
    pred_ = pred.view(size, -1)
    target_ = target.view(size, -1)
    intersection = (pred_ * target_).sum(1)
    dice = (2. * intersection + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
    return 1 - dice.mean()

def multi_class_dice_loss(output, target):
    """output: [B, 2, H, W], target: [B, H, W] 值0/1"""
    fg_prob = F.softmax(output, dim=1)[:, 1]  # [B, H, W]
    return dice_loss(fg_prob, target.float())

def _single_loss(output, target, dice_weight: float = 2.0):
    """一次 CE + Dice，返回标量"""
    ce = F.cross_entropy(output, target.long())
    dice = multi_class_dice_loss(output, target)
    return ce + dice_weight * dice

# ========= 总损失类 ========= #
class TotalLoss(nn.Module):
    """
    深度监督 + 主输出统一计算。
    用法:
        criterion = TotalLoss(ds_weights=[0.2, 0.3, 0.4, 0.5], dice_weight=2.0)
        loss = criterion((ds4, ds3, ds2, ds1), main_out, target)
    """
    def __init__(self, ds_weights=None, dice_weight: float = 2.0):
        super().__init__()
        self.ds_weights = ds_weights or [0.2, 0.3, 0.4, 0.5]
        self.dice_weight = dice_weight
        assert len(self.ds_weights) == 4, "需要4个深度监督权重"

    def forward(self, ds_outputs, main_output, target):
        """
        ds_outputs: tuple(ds4, ds3, ds2, ds1)  对应 1/16,1/8,1/4,1/2 分辨率
        main_output: [B, 2, H, W]
        target:      [B, H, W]  值0/1
        """
        # 主输出
        main_loss = _single_loss(main_output, target, self.dice_weight)

        # 深度监督
        ds_loss = 0.0
        for out, w in zip(ds_outputs, self.ds_weights):
            ds_loss += w * _single_loss(out, target, self.dice_weight)

        return main_loss + ds_loss





