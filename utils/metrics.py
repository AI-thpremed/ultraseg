# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
#
# def compute_metrics(pred, mask, num_classes, ignore_index=None):
#     """
#     pred   : (B,C,H,W) softmax logits
#     mask   : (B,H,W)   long
#     returns dict: {
#         'iou_cls':[c], 'dice_cls':[c],
#         'miou_with_bg':float, 'miou_no_bg':float,
#         'mdice_with_bg':float,'mdice_no_bg':float
#     }
#     """
#     pred = torch.argmax(pred, dim=1).flatten()   # (N,)
#     mask = mask.flatten()
#
#     if ignore_index is not None:
#         keep = mask != ignore_index
#         pred, mask = pred[keep], mask[keep]
#
#     ious, dices = [], []
#     for cls in range(num_classes):
#         pred_cls = pred == cls
#         mask_cls = mask == cls
#         inter = (pred_cls & mask_cls).sum().float()
#         union = (pred_cls | mask_cls).sum().float()
#         iou = (inter / (union + 1e-7)).item()
#         dice = (2 * inter / (pred_cls.sum() + mask_cls.sum() + 1e-7)).item()
#         ious.append(iou)
#         dices.append(dice)
#
#     metrics = {
#         'iou_cls': ious,
#         'dice_cls': dices,
#         'miou_with_bg': torch.tensor(ious).mean().item(),
#         'miou_no_bg': torch.tensor(ious[1:]).mean().item() if num_classes > 1 else None,
#         'mdice_with_bg': torch.tensor(dices).mean().item(),
#         'mdice_no_bg': torch.tensor(dices[1:]).mean().item() if num_classes > 1 else None,
#     }
#     return metrics


from scipy.spatial.distance import directed_hausdorff
import numpy as np
from sklearn.metrics import precision_score, recall_score


def compute_plus(pred, mask, num_classes, ignore_index=None):
    """
    pred: (B,H,W) 预测标签
    mask: (B,H,W) 真实标签
    """
    B = pred.shape[0]
    batch_haus, batch_prec, batch_rec = [], [], []

    for i in range(B):
        pred_single = pred[i]  # (H,W)
        mask_single = mask[i]  # (H,W)

        if ignore_index is not None:
            keep = mask_single != ignore_index
            pred_single = pred_single[keep]
            mask_single = mask_single[keep]

        # 处理空标签
        if pred_single.numel() == 0 or mask_single.numel() == 0:
            batch_haus.append(float('inf'))
            # 对于空标签，precision/recall 设为 0
            batch_prec.append(0.0)
            batch_rec.append(0.0)
            continue

        try:
            # 对单个样本计算 Hausdorff
            pred_coords = torch.stack(torch.where(pred_single), dim=1).cpu().numpy()  # (N1,2)
            mask_coords = torch.stack(torch.where(mask_single), dim=1).cpu().numpy()  # (N2,2)

            if len(pred_coords) == 0 or len(mask_coords) == 0:
                batch_haus.append(float('inf'))
            else:
                haus_list = [
                    directed_hausdorff(pred_coords, mask_coords)[0],
                    directed_hausdorff(mask_coords, pred_coords)[0]
                ]
                batch_haus.append(float(max(haus_list)))
        except:
            batch_haus.append(float('inf'))

        # 对单个样本计算 Precision/Recall
        try:
            pred_flat = pred_single.cpu().numpy().ravel()
            mask_flat = mask_single.cpu().numpy().ravel()
            prec = precision_score(mask_flat, pred_flat, average='macro', zero_division=0)
            rec = recall_score(mask_flat, pred_flat, average='macro', zero_division=0)
            batch_prec.append(prec)
            batch_rec.append(rec)
        except:
            batch_prec.append(0.0)
            batch_rec.append(0.0)

    # 对 batch 求平均，处理可能的 inf 值
    haus_vals = [x for x in batch_haus if x != float('inf')]
    avg_haus = np.mean(haus_vals) if haus_vals else float('inf')

    return {
        'hausdorff95': avg_haus,
        'precision': np.mean(batch_prec),
        'recall': np.mean(batch_rec)
    }
# 修复验证阶段的初始化


def compute_metrics(pred, mask, num_classes, ignore_index=None):
    """
    pred: (B,C,H,W) softmax logits
    mask: (B,H,W) long
    """
    B, C, H, W = pred.shape
    pred_label = torch.argmax(pred, dim=1)  # (B,H,W)

    flat_pred = pred_label.view(-1)
    flat_mask = mask.view(-1)

    if ignore_index is not None:
        keep = flat_mask != ignore_index
        flat_pred, flat_mask = flat_pred[keep], flat_mask[keep]

    # 1. IoU/Dice 逐类（全局计算，不是逐样本）
    ious, dices = [], []
    for cls in range(num_classes):
        pred_cls = (flat_pred == cls)
        mask_cls = (flat_mask == cls)

        inter = (pred_cls & mask_cls).sum().float()
        union = (pred_cls | mask_cls).sum().float()
        pred_sum = pred_cls.sum().float()
        mask_sum = mask_cls.sum().float()

        iou = (inter / (union + 1e-7)).item()
        dice = (2 * inter / (pred_sum + mask_sum + 1e-7)).item()

        ious.append(iou)
        dices.append(dice)

    metrics = {
        'iou_cls': ious,
        'dice_cls': dices,
        'miou_with_bg': torch.tensor(ious).mean().item(),
        'mdice_with_bg': torch.tensor(dices).mean().item(),
    }

    if num_classes > 1:
        metrics['miou_no_bg'] = torch.tensor(ious[1:]).mean().item()
        metrics['mdice_no_bg'] = torch.tensor(dices[1:]).mean().item()
    else:
        metrics['miou_no_bg'] = None
        metrics['mdice_no_bg'] = None

    # 2. Hausdorff/PR（逐样本计算）
    plus_metrics = compute_plus(pred_label, mask, num_classes, ignore_index)
    metrics.update(plus_metrics)

    return metrics

