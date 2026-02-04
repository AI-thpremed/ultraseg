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

import os, json, torch
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import random
import cv2
import torchvision.transforms as T

class SegDataset(Dataset):
    IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    def __init__(self,
                 img_dir,
                  mask_dir,config,
                  img_size=512,
                 val=False):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.img_size = img_size
        self.val      = val
        self.config=config

        self.ids = [p.stem for p in self.img_dir.iterdir()
                    if p.suffix.lower() in self.IMG_EXTS]

        #  [0,1]
        self.to_tensor = T.Compose([T.ToTensor()])

        # if self.clahe:
        self.clahe_tf = A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)

    def _augment_train(self, image, mask):
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        if self.config["training"]["data_augmentation"]["rotate"]["enable"]:
            angle = random.randint(self.config["training"]["data_augmentation"]["rotate"]["range"][0],
                                   self.config["training"]["data_augmentation"]["rotate"]["range"][1])
            center = (self.img_size // 2, self.img_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (self.img_size, self.img_size),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)

        if self.config["training"]["data_augmentation"]["flip"]:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

        if self.config["training"]["data_augmentation"]["clahe"]:
            image = self.clahe_tf(image=image)['image']

        if self.config["training"]["data_augmentation"]["random_brightness"] or \
           self.config["training"]["data_augmentation"]["random_contrast"] or \
           self.config["training"]["data_augmentation"]["random_blur"]:
            aug = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5) if self.config["training"]["data_augmentation"]["random_brightness"] else A.NoOp(),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5) if self.config["training"]["data_augmentation"]["random_contrast"] else A.NoOp(),
                A.Blur(blur_limit=3, p=0.3) if self.config["training"]["data_augmentation"]["random_blur"] else A.NoOp(),
            ])
            augmented = aug(image=image)
            image = augmented['image']

        return image, mask

    def _augment_val(self, image, mask=None):
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (self.img_size, self.img_size),
                              interpolation=cv2.INTER_NEAREST)

        #  CLAHE
        if self.config["validation"]["data_augmentation"]["clahe"]:
            image = self.clahe_tf(image=image)['image']

        return image, mask




    def __len__(self):
        return len(self.ids)


    def __getitem__(self, idx):
        name = self.ids[idx]
        img_path = next(self.img_dir.glob(f'{name}.*'))

        try:
            image = cv2.imread(str(img_path))[:, :, ::-1]  # BGR→RGB

            original_height, original_width = image.shape[:2]

            mask = None
            if self.mask_dir:
                mask_path = self.mask_dir / f'{name}.png'
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if self.val:
                image, mask = self._augment_val(image, mask)
            else:
                image, mask = self._augment_train(image, mask)

            image = self.to_tensor(image)
            if mask is not None:
                mask = torch.from_numpy(mask).long()
            else:
                mask = torch.zeros((self.img_size, self.img_size), dtype=torch.long)  # 默认全零掩码
            size_tensor = torch.tensor([original_height, original_width], dtype=torch.long)

            return image, mask,name,size_tensor
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None,None  # 返回 None 以便后续处理
