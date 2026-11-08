"""Dense clear-weather pseudo-labels with Depth Anything V2 (Large), GPU.

For every training frame: DAv2 relative disparity on the clear image, aligned to the
sparse LiDAR in disparity space (least squares, one outlier-rejection pass), converted
to metric depth, cropped to the KITTI benchmark window (352x1216) and stored as
uint16 PNG (depth * 256). Resumable: existing outputs are skipped.

Usage: python gen_pseudo.py [--shard i --num_shards n] [--limit k]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import KB_H, KB_W, ROOT, kb_crop_box, save_json  # noqa: E402

DAV2 = f"{ROOT}/assets/checkpoints/Depth-Anything-V2-Large-hf"
MIN_D, MAX_D = 1e-3, 80.0


def align_disparity(pred_disp, lidar):
    valid = (lidar > MIN_D) & (lidar < MAX_D)
    x, y = pred_disp[valid], 1.0 / lidar[valid]
    A = np.stack([x, np.ones_like(x)], 1)
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    res = np.abs(A @ np.array([s, t]) - y)
    keep = res < 3 * np.median(res) + 1e-9
    s, t = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
    disp = np.clip(s * pred_disp + t, 1.0 / MAX_D, 1.0 / MIN_D)
    return 1.0 / disp, float(s), float(t), int(valid.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(DAV2)
    model = AutoModelForDepthEstimation.from_pretrained(DAV2).to(args.device).eval()
    if args.device == "cuda":
        model = model.half()

    frames = json.load(open(f"{ROOT}/revision/manifests/train.json"))["frames"]
    frames = frames[args.shard::args.num_shards]
    if args.limit:
        frames = frames[: args.limit]
    stats = []
    for fr in tqdm(frames, desc=f"DAv2 shard {args.shard}"):
        out = fr["pseudo"]
        if os.path.exists(out):
            continue
        img = Image.open(fr["rgb"]).convert("RGB")
        lidar = np.asarray(Image.open(fr["lidar"]), dtype=np.float32) / 256.0
        inputs = proc(images=img, return_tensors="pt").to(args.device)
        if args.device == "cuda":
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.no_grad():
            disp = model(**inputs).predicted_depth  # relative inverse depth
        disp = F.interpolate(disp[:, None].float(), size=lidar.shape, mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
        depth, s, t, n = align_disparity(disp, lidar)
        top, left = kb_crop_box(*depth.shape)
        crop = depth[top:top + KB_H, left:left + KB_W]
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(np.round(crop * 256).astype(np.uint16)).save(out)
        stats.append({"id": fr["id"], "scale": s, "shift": t, "lidar_points": n})
    save_json(stats, f"{ROOT}/revision/logs/pseudo_stats_shard{args.shard}.json")


if __name__ == "__main__":
    main()
