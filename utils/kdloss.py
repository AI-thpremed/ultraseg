import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.losses import  CrossEntropyLoss2d






class KDLoss(nn.Module):
    def __init__(self, T=3.0, alpha=0.5, weight=None, ignore_index=255):
        super().__init__()
        self.T = T
        self.alpha = alpha
        # 分割损失：用你的 2D CrossEntropy
        self.ce = CrossEntropyLoss2d(weight=weight, ignore_index=ignore_index)

    def forward(self, student_logit, teacher_logit, target):
        """
        student_logit: [B,2,H,W]
        teacher_logit:[B,2,H,W] (已detach)
        target:       [B,H,W]   0/1 整数
        """
        # 1. 分割损失
        loss_seg = self.ce(student_logit, target)

        # 2. KD 损失：KL(student||teacher)  双通道 softmax
        with torch.no_grad():
            t_prob = F.softmax(teacher_logit / self.T, dim=1)
        s_logp = F.log_softmax(student_logit / self.T, dim=1)
        loss_kd = F.kl_div(s_logp, t_prob, reduction='batchmean') * (self.T ** 2)

        return self.alpha * loss_seg + (1 - self.alpha) * loss_kd, \
               {'loss_seg': loss_seg.item(), 'loss_kd': loss_kd.item()}