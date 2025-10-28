from torchvision.models.segmentation import deeplabv3_resnet50
import torch
from utils.model import SegFormerTorchvisionCompact
from transformers import SegformerForSemanticSegmentation
import numpy as np

# Cityscapes colormap
CITYSCAPES_COLORS = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32]
], dtype=np.uint8)

def colorize(mask):
    return CITYSCAPES_COLORS[mask]

def load_model(ckpt_path, model_type, device="cuda"):
    if model_type == 'deeplabv3':
        model = deeplabv3_resnet50(weights=None, num_classes=19)
    elif model_type == 'segformer':
        base = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=19,
            ignore_mismatched_sizes=True
        )
        model = SegFormerTorchvisionCompact(base)
    else:
        raise NameError('Model Type is not registered')

    # Load checkpoint on CPU for safety
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Resolve state dict
    state = (
        ckpt.get("model_state_dict")
        or ckpt.get("model_state")
        or ckpt.get("state_dict")
        or (ckpt if isinstance(ckpt, dict) else None)
    )
    if state is None:
        raise RuntimeError("No state_dict found in checkpoint")

    # Load weights into model
    missing = model.load_state_dict(state, strict=False)
    print(">> missing keys:", missing.missing_keys)
    print(">> unexpected keys:", missing.unexpected_keys)

    # Move to device and eval mode
    model.to(device)
    model.eval()
    return model