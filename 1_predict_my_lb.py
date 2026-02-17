import os
import sys
import json
import random
import numpy as np
import pandas as pd
import datetime
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn
from tqdm import tqdm
# from prepare_masks import prepare_masks_and_mapping_mask
from utils.metrics import compute_metrics
from utils.losses import CE_DiceLoss, LovaszSoftmax, FocalLoss,DynamicConnectivityLoss
from config_manager import ConfigManager
import random
import numpy as np
import time
import cv2
from makedata.lb_utils import GT_BceDiceLoss,GT_BceDiceLoss_Mine
from makedata.dataset import SegDataset
from new_models.ultraseg108k import UltraSeg108
from new_models.ultraseg130k import UltraSeg130


class TqdmToLog:
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.log_func(line.rstrip())

    def flush(self):
        pass


def setup_logging(res_dir):
    """设置日志文件"""
    log_file_path = res_dir / 'training.log'
    log_file = open(log_file_path, 'a', encoding='utf-8')

    def log(msg, level="INFO"):
        if msg is None:
            if log_file:
                log_file.close()
            return

        print(msg)
        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

    return log, log_file


def predict_mask_segmentation(cfg):
    config_manager = ConfigManager(config_path="G:\Github-folder\UltraSeg\ultraseg\config.json")
    config = config_manager.config
    image_size = 256
    seed =cfg['seed']

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    root_cache = Path(__file__).parent / 'cache'
    root_results = Path(__file__).parent / 'results'
    root_cache.mkdir(exist_ok=True)
    root_results.mkdir(exist_ok=True)
    name1=cfg['dataset']
    name2=cfg['algo']
    name3=seed
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    res_dir = root_results / f'0PREDICT_{name1}_{name2}_{name3}_train_{timestamp}'
    res_dir.mkdir(exist_ok=True)

    (res_dir / 'predict').mkdir(exist_ok=True)
    (res_dir / 'predict_orgsize').mkdir(exist_ok=True)

    cfg['results'] = str(res_dir)

    cfg_save_path = res_dir / 'training_config.json'
    with open(cfg_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    log, log_file = setup_logging(res_dir)

    train_loss_df = pd.DataFrame(columns=['epoch', 'train_loss'])
    val_loss_df = pd.DataFrame(columns=['epoch', 'val_loss'])

    try:
        log("Starting training process...")

        log("!!seed=" + str(seed))

        # 加载标签映射
        mapping_path =  cfg['label_map']
        with open(mapping_path) as f:
            label_map = json.load(f)
        class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        num_classes = len(class_names)

        num_workers = 2

        val_ds = SegDataset(cfg['test_img'],  cfg['test_mask'], config, image_size, val=True)



        val_dl = DataLoader(val_ds, batch_size=4,
                            shuffle=False, num_workers=num_workers, pin_memory=False)

        device = torch.device("cuda:"+str(cfg['gpunum']) if torch.cuda.is_available() else "cpu")
        log(f"Using device: {device}")

      
        log(f"Using model: {cfg['algo']}")

        if  cfg['algo'] =="ultraseg108":
            model = UltraSeg108(in_ch=3, out_ch=num_classes,key=3).to(device)
            criterion = GT_BceDiceLoss_Mine(wb=1, wd=1)
        elif cfg['algo'] =="ultraseg130":
            model = UltraSeg130(in_ch=3, out_ch=num_classes,key=3).to(device)
            criterion = GT_BceDiceLoss_Mine(wb=1, wd=1)

        best_path = Path(cfg['modeldir']) / 'best.pth'

        ckpt = torch.load(best_path, map_location=device)

        model.load_state_dict(ckpt)


        log(f"Using model: {cfg['algo']}")


        criterion = GT_BceDiceLoss(wb=1, wd=1)


        log(f"Using loss function: {cfg['loss']}")

        evemetric = config["training"].get("evemetric", "iou")


        lr =  3e-4
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        log(f"learning rate:{lr}")

        # 早停设置
        earlystop = config["training"].get("earlystop", 0)
        if not isinstance(earlystop, int):
            try:
                earlystop = int(earlystop)
            except (ValueError, TypeError):
                earlystop = 0
                log(f"Warning: earlystop value '{earlystop}' is not valid. Using default value 0.")

        if earlystop > cfg['epochs'] or earlystop < 0:
            earlystop = 0

        best_miou_no_bg = 0.0
        best_mdice_no_bg = 0.0
        early_stop_counter = 0

        log(f"Starting training for {cfg['epochs']} epochs")

        metrics_sum = {
            'iou_cls': [0] * num_classes,
            'dice_cls': [0] * num_classes,
            'miou_with_bg': 0,
            'miou_no_bg': 0,
            'mdice_with_bg': 0,
            'mdice_no_bg': 0,
            'hausdorff95': 0,
            'precision': 0,
            'recall': 0
        }


        # 验证阶段
        model.eval()
        val_loss, count = 0, 0
        total_val_batches = len(val_dl)
        epoch=0
        pbar = tqdm(val_dl, desc=f'Epoch {epoch + 1}/{cfg["epochs"]} [VAL]')



        with torch.no_grad():
            for batch_idx, (img, msk,name, original_sizes) in enumerate(pbar):
                img, msk = img.to(device), msk.to(device)

                # pred = model(img)
                gt_pre,key_points, pred = model(img)

                count += 1
                batch_met = compute_metrics(pred, msk, num_classes)

                # 累加逐类指标
                for k in ('iou_cls', 'dice_cls'):
                    metrics_sum[k] = [a + b for a, b in zip(metrics_sum[k], batch_met[k])]
                # 累加全局指标（含新增 3 项）
                for k in ('miou_with_bg', 'miou_no_bg', 'mdice_with_bg', 'mdice_no_bg',
                            'hausdorff95', 'precision', 'recall'):
                    if batch_met[k] is not None:
                        metrics_sum[k] += batch_met[k]

                pbar.set_postfix(loss=f'{val_loss / count:.4f}')
                if (batch_idx + 1) % max(1, total_val_batches // 10) == 0 or batch_idx == total_val_batches - 1:
                    progress = (batch_idx + 1) / total_val_batches * 100
                    log(f'Val Epoch {epoch + 1} | {batch_idx + 1}/{total_val_batches} batches ({progress:.0f}%)')

                pred = pred.argmax(1).cpu().numpy().astype(np.uint8)  # (B,H,W)
                for b in range(pred.shape[0]):
                    save_path = res_dir / 'predict' / f'{name[b]}.png'
                    cv2.imwrite(str(save_path), pred[b])      # 尺寸保持 256×256



        val_loss /= count
        for k in ('iou_cls', 'dice_cls'):
            metrics_sum[k] = [v / count for v in metrics_sum[k]]
        for k in ('miou_with_bg', 'miou_no_bg', 'mdice_with_bg', 'mdice_no_bg',
                    'hausdorff95', 'precision', 'recall'):
            if isinstance(metrics_sum[k], (int, float)):
                metrics_sum[k] /= count

        log_str = f"Epoch {epoch + 1:03d} | val_loss {val_loss:.4f}\n"
        new_row = pd.DataFrame({'epoch': [epoch + 1], 'val_loss': [val_loss]})
        val_loss_df = pd.concat([val_loss_df, new_row], ignore_index=True)

        for cls, iou, dice in zip(class_names, metrics_sum['iou_cls'], metrics_sum['dice_cls']):
            log_str += f"  {cls:<10} IoU={iou:.4f} Dice={dice:.4f}\n"
        log_str += (f"  Mean(w/ bg) IoU={metrics_sum['miou_with_bg']:.4f} "
                    f"Dice={metrics_sum['mdice_with_bg']:.4f}\n")
        if metrics_sum['miou_no_bg'] is not None:
            log_str += (f"  Mean(no bg) IoU={metrics_sum['miou_no_bg']:.4f} "
                        f"Dice={metrics_sum['mdice_no_bg']:.4f}\n")

        # 新增 3 行打印
        log_str += (f"  Hausdorff@95={metrics_sum['hausdorff95']:.2f}px  "
                    f"Precision={metrics_sum['precision']:.3f}  "
                    f"Recall={metrics_sum['recall']:.3f}\n")

        log(log_str)


        log("Predict completed successfully!")

    except Exception as e:
        log(f"Predict error: {e}")
        raise
    finally:
        log(None)
        if log_file:
            log_file.close()




def main():
    parser = argparse.ArgumentParser(description='Train image segmentation model')

    parser.add_argument('--dataset', type=str, default='bkai-igh',
                        help='CVC Kvasir PolypDB PolypGen21  ETIS')
    parser.add_argument('--label_map', type=str, default='/home/ubuntu/Desktop/dataset/label_mapping.json')
    parser.add_argument('--algo', type=str, default='ultraseg130', help='ultraseg108，ultraseg130')
    parser.add_argument('--loss', type=str,default='CrossEntropyLoss', choices=[
        'CrossEntropyLoss', 'CE_DiceLoss', 'FocalLoss', 'LovaszSoftmax'
    ], help='Loss function')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--gpunum',default=0, type=int)
    parser.add_argument('--seed', type=int, default=40)
    args = parser.parse_args()
    modeldir=f'/home/ubuntu/Desktop/gaowh/cmd-seg/results/ssl_develop_{args.algo}_{args.seed}'

    cfg = {
        'dataset': args.dataset,
        'train_img': f'/home/ubuntu/Desktop/dataset/{args.dataset}/train/images',
        'test_img': f'/home/ubuntu/Desktop/dataset/{args.dataset}/images',
        'train_mask': f'/home/ubuntu/Desktop/dataset/{args.dataset}/train/masks',
        'test_mask': f'/home/ubuntu/Desktop/dataset/{args.dataset}/masks',
        'label_map':args.label_map,
        'algo': args.algo,
        'loss': args.loss,
        'epochs': args.epochs,
        'seed': args.seed,
        'gpunum':args.gpunum,
        'modeldir': modeldir

    }

    print("Starting training with configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    predict_mask_segmentation(cfg)


if __name__ == "__main__":
    main()

    