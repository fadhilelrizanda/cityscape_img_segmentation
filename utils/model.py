from torch.amp import GradScaler, autocast
import torch
from tqdm.auto import tqdm
import os
import torch.nn.functional as F

class SegFormerTorchvisionCompact(torch.nn.Module):  # name can stay Compact if you prefer
    def __init__(self, hf_model, normalize=False):
        super().__init__()
        self.net = hf_model
        self.normalize = normalize
        if normalize:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
            self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        out = self.net(pixel_values=x)
        # NOTE: use the tuple (H, W), not a scalar
        logits = F.interpolate(out.logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return {"out": logits}


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    total = 0; running = 0.0
    train_step = tqdm(enumerate(train_loader),total=len(train_loader),desc="Training Step",leave=False)
    for i, (imgs,masks) in train_step:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast('cuda'):
            out = model(imgs)["out"]                 # (B, C, H, W)
            loss = criterion(out, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * imgs.size(0); total += imgs.size(0)
    return running/total

def evaluate(model, val_loader, criterion, device, num_classes=19, ignore_index=255):
    model.eval()
    total = 0
    running = 0.0

    # intersection and union per class
    inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=device)

    val_step = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validation Step", leave=False)
    for i, (imgs, masks) in val_step:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        with torch.no_grad():
            out = model(imgs)["out"]  # [B,C,H,W]
            if out.shape[-2:] != masks.shape[-2:]:
                out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            loss = criterion(out, masks)
            running += loss.item() * imgs.size(0)
            total += imgs.size(0)

            preds = out.argmax(dim=1)  # [B,H,W]

            # mask out ignore_index
            mask_valid = masks != ignore_index
            preds = preds[mask_valid]
            labels = masks[mask_valid]

            # accumulate intersection and union per class
            for cls in range(num_classes):
                pred_inds = preds == cls
                target_inds = labels == cls
                if target_inds.sum() == 0 and pred_inds.sum() == 0:
                    continue
                inter[cls] += (pred_inds & target_inds).sum()
                union[cls] += (pred_inds | target_inds).sum()

    val_loss = running / total
    iou = inter / (union + 1e-6)
    miou = iou.mean().item()

    return val_loss, miou

def save_checkpoint(model, optimizer, scaler, epoch, best_val, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val,
    }, path)