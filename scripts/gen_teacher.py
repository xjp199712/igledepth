"""Clear-weather self-distillation targets: the frozen base's prediction on the clear KB crop.

The prediction is already in the base's affine-invariant [0,1] output space, so it is
stored as uint16 PNG (value * 65535) under revision/<out_dir>/<rel>. The student is then
trained to reproduce this clear-image prediction from every weather rendering of the same
frame (train_rev.py --target teacher). Resumable.

Usage (paper settings):
  python gen_teacher.py --steps 1 --batch 8                                   # Marigold -> teacher_mg
  python gen_teacher.py --base e2eft --single_step --proc_w 768 --out_dir teacher_e2eft --batch 8
  python gen_teacher.py --base lotus_g --single_step --proc_w 768 --out_dir teacher_lotusg --batch 8
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (KB_H, KB_W, ROOT, DepthModel, kb_crop_box, proc_size, read_rgb,  # noqa: E402
                        resize_t, to_tensor_rgb)


def teacher_path(fr, out_dir="teacher_mg"):
    return fr["pseudo"].replace("pseudo_dav2", out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base", default="marigold")
    ap.add_argument("--single_step", action="store_true")
    ap.add_argument("--out_dir", default="teacher_mg")
    ap.add_argument("--proc_w", type=int, default=0, help="predict (and store) at this width")
    args = ap.parse_args()
    dev = "cuda"
    model = DepthModel("marigold", base=args.base, single_step=args.single_step).to(dev).eval()

    frames = json.load(open(f"{ROOT}/revision/manifests/train.json"))["frames"]
    frames = [f for f in frames if os.path.exists(f["pseudo"]) and not os.path.exists(teacher_path(f, args.out_dir))]
    if args.limit:
        frames = frames[: args.limit]
    print(f"to do: {len(frames)}")
    for k in tqdm(range(0, len(frames), args.batch), desc="teacher"):
        chunk = frames[k:k + args.batch]
        xs = []
        for fr in chunk:
            arr = read_rgb(fr["rgb"])
            top, left = kb_crop_box(*arr.shape[:2])
            xs.append(resize_t(to_tensor_rgb(arr[top:top + KB_H, left:left + KB_W]),
                               proc_size(KB_H, KB_W, args.proc_w)))
        g = torch.Generator(device=dev).manual_seed(k)
        with torch.no_grad():
            pred = model.predict(torch.stack(xs).to(dev), steps=args.steps, generator=g)
        pred = pred[:, 0].float().clamp(0, 1).cpu().numpy()
        for fr, p in zip(chunk, pred):
            out = teacher_path(fr, args.out_dir)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            Image.fromarray(np.round(p * 65535).astype(np.uint16)).save(out)


if __name__ == "__main__":
    main()

