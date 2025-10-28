import torch
from torch.utils.data import DataLoader
import os
from utils.preprocessing import CityscapesAlb, train_transform, val_transform_full
from utils.model import train_one_epoch, evaluate, save_checkpoint,SegFormerTorchvisionCompact
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.amp import GradScaler
from dotenv import load_dotenv  
import wandb  
import yaml
import argparse
from tqdm.auto import tqdm
from torchvision.models import ResNet50_Weights
from transformers import SegformerForSemanticSegmentation, AutoConfig


# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
load_dotenv("myenv.env")
wandb_key = os.getenv("WANDB_API_KEY")
wandb.login(key=wandb_key)

def read_yaml(path):
    with open(path, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
        return data

def main(cfg):
    config = read_yaml(cfg)
    
    wandb.init(
        project="cityscapes-img_seg",  # change to your project name
        name=config['model']['name'],  # run name
        config={
            "model": config['model']['name'],
            "lr": config['optim']['lr'],
            "weight_decay": config['optim']['weight_decay'],
            "epochs": config['train']['epochs'],
            "batch_size": config['data']['batch_size']
        }
    )
    
    ROOT = "./dataset"   # <-- change this
    train_set = CityscapesAlb(ROOT, split="train", transform=train_transform)
    val_set   = CityscapesAlb(ROOT, split="val",   transform=val_transform_full)
    train_loader = DataLoader(train_set, batch_size=config['data']['batch_size']
                              , shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=2
                              , shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config['model']['num_classes']


    if config['model']['arch'] == 'deeplabv3':
        if config['model']['pretrained']:
            model = deeplabv3_resnet50(
            weights=None, 
            weights_backbone=ResNet50_Weights.IMAGENET1K_V2,  
            num_classes=num_classes).to(device)

        else:
            model = deeplabv3_resnet50(weights=config['model']['weight'], num_classes=num_classes).to(device)

    elif config['model']['arch'] == 'SegFormer':
        
        if config['model']['weight']:
            base = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=config['model']['num_classes'],
                ignore_mismatched_sizes=True
            )
            model = SegFormerTorchvisionCompact(base).to(device)

        else:
            cfg = AutoConfig.from_pretrained("nvidia/mit-b2")   # get architecture config only
            cfg.num_labels = config['model']['num_classes']
            cfg.semantic_loss_ignore_index = 255
            base = SegformerForSemanticSegmentation(cfg)  # <- random init
            model = SegFormerTorchvisionCompact(base, normalize=False).to(device)

    else:
        print("Model Arch is not recognized")
        
    criterion = nn.CrossEntropyLoss(ignore_index=255)  
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'], weight_decay=config['optim']['weight_decay'])
    
    scaler = GradScaler("cuda")
    wandb.watch(model, log="all", log_freq=100)
    best_miou  = -1
    best_epoch = -1
    patience = 10
    min_delta = 1e-4
    no_improve = 0

    
    for epoch in tqdm(range(config['train']['epochs']), desc=f"Training {config['model']['name']}",leave=False):  
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler,device)
        val_loss, miou = evaluate(model,val_loader, criterion,device)
        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val miou {miou:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mIoU": miou,
        })

        improved = miou > best_miou + min_delta
        if improved:
            best_miou = miou
            best_epoch = epoch
            ckpt_path = f"checkpoint/{config['model']['name']}/best_mIoU_epoch{epoch:02d}.pt"
            save_checkpoint(model, optimizer, scaler, epoch+1, best_miou, ckpt_path)
            no_improve = 0
        else:
            no_improve += 1
    
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no mIoU improvement for {patience} epochs).")
            break

    wandb.run.summary["best_mIoU"] = best_miou
    wandb.run.summary["best_epoch"] = best_epoch    
    wandb.finish()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Semantic Segmentation")
    parser.add_argument(
        "--config", type = str, help="path to yaml config file"
    )
    args = parser.parse_args()
    main(args.config)

