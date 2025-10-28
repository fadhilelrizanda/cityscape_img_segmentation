import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.datasets import Cityscapes
from PIL import Image
import torch


def make_labelid_to_trainid():
    lut = np.full(256, 255, dtype=np.uint8) 
    mapping = {
        7:0, 8:1, 11:2, 12:3, 13:4, 17:5, 19:6, 20:7,
        21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14,
        28:15, 31:16, 32:17, 33:18
    }
    for k,v in mapping.items():
        lut[k] = v
    return lut
LUT = make_labelid_to_trainid()


def mask_to_trainid(mask_pil):
    m = np.array(mask_pil, dtype=np.uint8)
    m = LUT[m]            
    return torch.from_numpy(m).long()  


train_transform = A.Compose(
    [
        A.RandomScale(scale_limit=(-0.5, 1.0), interpolation=1, p=1.0),  # 0.5x ~ 2.0x
        A.PadIfNeeded(min_height=512, min_width=1024, border_mode=0, p=1.0),
        A.RandomCrop(height=512, width=1024, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=7, p=1.0),
        ], p=0.2),
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.2), p=1.0),
            A.GaussNoise(std_range=(10/255.0, 50/255.0), p=1.0)
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),   
    ],
    additional_targets={},   
)

val_transform_full = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(transpose_mask=False),
    ]
)

class CityscapesAlb(Cityscapes):
    def __init__(self, root, split, transform):
        super().__init__(root=root, split=split, mode="fine", target_type="semantic",
                         transform=None, target_transform=None)
        self.tfm = transform

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)       
        img_np  = np.array(img)                     
        mask_np = np.array(mask, dtype=np.uint8)    
        mask_np = LUT[mask_np]                     
        augmented = self.tfm(image=img_np, mask=mask_np)
        img_t = augmented["image"]                  
        mask_t = augmented["mask"].long()          
        return img_t, mask_t

