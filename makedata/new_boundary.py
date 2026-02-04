import os, cv2, glob
import numpy as np
from tqdm import tqdm
import argparse

def mask2boundary(mask_path, save_path, dilate_k=5, gblur_ks=0):
    """
    0/1 mask -> 0/255 边缘图
    dilate_k: 边缘膨胀宽度
    gblur_ks: 高斯核大小，0 表示不做
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)   # 0-255 或 0-1 都会读到 0-255
    # 二值化：>0 即为病灶
    mask = (mask > 0).astype(np.uint8)          # 0/1
    if gblur_ks > 0:
        mask = cv2.GaussianBlur(mask*255, (gblur_ks, gblur_ks), 0)
    edge = cv2.Canny((mask*255).astype(np.uint8), 30, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    edge = cv2.dilate(edge, kernel, iterations=1)
    cv2.imwrite(save_path, edge)   # 保存为 0/255 单通道 png

def run(root, split_list=('train',), dilate_k=5):
    for split in split_list:
        mask_dir = os.path.join(root, split, 'masks')
        out_dir  = os.path.join(root, split, 'points_boundary2')
        os.makedirs(out_dir, exist_ok=True)
        files = sorted(glob.glob(os.path.join(mask_dir, '*.png')))
        for f in tqdm(files, desc=f'{split}-points'):
            save_name = os.path.basename(f)
            mask2boundary(f, os.path.join(out_dir, save_name),
                          dilate_k=dilate_k, gblur_ks=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/Desktop/dataset/WCEBleedGen')
    parser.add_argument('--dilate', type=int, default=5,
                        help='edge dilation width')
    args = parser.parse_args()
    run(args.data_root, dilate_k=args.dilate)