#!/usr/bin/env python3
import os, argparse, glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import time
from utils.inference import load_model,colorize

@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Video file or folder of frames")
    ap.add_argument("--output", default="seg_output.mp4", help="Output video file")
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model_type", default="deeplabv3")
    ap.add_argument("--size", type=int, nargs=2, default=[1024,2048], help="Model input H W")
    ap.add_argument("--fps", type=float, default=17.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="Overlay weight")
    args = ap.parse_args()

    device = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"
    model = load_model(args.checkpoint,args.model_type, device)

    # Frame iterator
    if os.path.isdir(args.input):
        frame_paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
        frames = [cv2.imread(p) for p in frame_paths]
    else:
        cap = cv2.VideoCapture(args.input)
        frames = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frames.append(f)
        cap.release()

    if not frames:
        raise RuntimeError("No frames found.")

    Hout, Wout = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (Wout, Hout))

    fps_list = []
    for f in tqdm(frames,desc="Predicting..."):
        t0 = time.time()
        # preprocess
        inp = cv2.cvtColor(cv2.resize(f, (args.size[1], args.size[0])), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(inp).permute(2, 0, 1).float() / 255.0          # (3,H,W)
        tensor = tensor.unsqueeze(0).to(device)                                   # (1,3,H,W)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        # forward
        out = model(tensor)["out"]
        if out.shape[-2:] != tensor.shape[-2:]:
            out = F.interpolate(out, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        pred = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


        # end timing
        t1 = time.time()
        fps_list.append(1.0 / (t1 - t0 + 1e-8))

        # overlay mask
        mask = cv2.resize(colorize(pred), (Wout, Hout), interpolation=cv2.INTER_NEAREST)
        overlay = cv2.addWeighted(f, 1 - args.alpha, mask[:, :, ::-1], args.alpha, 0)

        # draw fps on frame
        inst_fps = fps_list[-1]
        cv2.putText(overlay, f"FPS: {inst_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(overlay)

    writer.release()
    print(f"Saved → {args.output}")
    print(f"Avg FPS : {sum(fps_list)/len(fps_list)}")

if __name__ == "__main__":
    main()
