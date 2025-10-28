# 🏙️ Cityscapes Semantic Segmentation — DeepLabV3 vs SegFormer

This project compares **[DeepLabV3](https://learnopencv.com/deeplabv3-ultimate-guide/)** and **[SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer)** on the [Cityscapes](https://www.cityscapes-dataset.com/) dataset for semantic segmentation, focusing on training performance, mIoU, and model efficiency.

---

## 📊 Project Overview

- **Dataset:** Cityscapes (gtFine + leftImg8bit)
- **Models:** DeepLabV3 (CNN) vs SegFormer (Transformer)
- **Goal:** Compare convergence speed, accuracy, and complexity
- **Metrics:** Loss & mIoU (tracked with [W&B](https://wandb.ai/))

| Model     | Best Epoch | mIoU  | Note                     |
| --------- | ---------- | ----- | ------------------------ |
| DeepLabV3 | 74         | 0.726 | Lightweight, efficient   |
| SegFormer | 49         | 0.727 | Slightly better, heavier |

---

## 🧰 Training Pipeline

**Augmentation (Albumentations):**

```python
A.Compose([
    A.RandomScale(scale_limit=(-0.5, 1.0), p=1.0),
    A.PadIfNeeded(512, 1024, border_mode=0),
    A.RandomCrop(512, 1024),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
    A.OneOf([A.GaussianBlur((3, 7)), A.MotionBlur(7)], p=0.2),
    A.OneOf([A.ISONoise(), A.GaussNoise()], p=0.2),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])
```

## Config

Example Config:

```
model:
  arch: "deeplabv3"
  num_classes: 19
  pretrained: True
optim:
  name: "AdamW"
  lr: 1e-5
train:
  epochs: 80
```

## Training

`python train.py --config ./path/to/config.yaml`

## Instalation

```
git clone https://github.com/fadhilelrizanda/cityscape_img_segmentation.git
cd cityscape_img_segmentation
pip install -r requirements.txt
```

## Result

![Segmentation Output](./demo.gif)
