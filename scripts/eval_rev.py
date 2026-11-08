"""Inference + metrics on WeatherKITTI (author-697 LiDAR GT), DrivingStereo and CADC.

One protocol for every model: per-image least-squares scale/shift alignment in depth,
depth cap 80 m, Eigen crop on KITTI. Writes per-image metrics (for paired tests) and a
summary. Optionally dumps SVDF kernel / IGLEM routing maps (--save_maps) for the
mechanism figure.

  python eval_rev.py --mode igle --ckpt runs/igledepth_v1/final_trainable.pt --tag igle_v1
  python eval_rev.py --mode marigold --tag marigold_zs            # zero-shot, no ckpt
  python eval_rev.py ... --steps 1 --tag igle_v1_1step            # single-step (E2E-FT style)
  python eval_rev.py ... --seed 1 --tag igle_v1_s1                # noise-seed study
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (KB_H, KB_W, ROOT, DepthModel, kb_crop_box, pad_to_multiple,  # noqa: E402
                        proc_size, read_rgb, resize_t, save_json, to_tensor_rgb)

MAN = f"{ROOT}/revision/manifests"
MAX_D = 80.0


def load_gt(s):
    d = s["depth"]
    if d["format"] == "npy_2d":
        return np.load(d["path"]).astype(np.float32)
    if d["format"] == "png_u16_div256":
        return np.asarray(Image.open(d["path"]), dtype=np.float32) / 256.0
    if d["format"] == "npy_array_index":
        arr = load_gt.cache.get(d["path"])
        if arr is None:
            arr = load_gt.cache[d["path"]] = np.load(d["path"], mmap_mode="r")
        return np.asarray(arr[d["index"]], dtype=np.float32)
    raise ValueError(d["format"])


load_gt.cache = {}


def prepare(s, set_name):
    """Returns (input tensor [-1,1], gt, eval_mask, region) where region tells how to map
    the prediction back onto the GT grid."""
    gt = load_gt(s)
    H, W = gt.shape
    if set_name.startswith("kitti"):
        arr = read_rgb(s["rgb_path"], size_wh=(W, H))
        top, left = kb_crop_box(H, W)
        inp = to_tensor_rgb(arr[top:top + KB_H, left:left + KB_W])
        mask = np.zeros_like(gt, bool)  # Eigen crop
        mask[int(0.3324324 * H):int(0.91351351 * H), int(0.0359477 * W):int(0.96405229 * W)] = True
        region = ("crop", top, left)
    elif set_name.startswith("cadc"):
        arr = read_rgb(s["rgb_path"], size_wh=(W, H))
        top, bottom = 232, 776  # crop used by the original project code
        inp = to_tensor_rgb(arr[top:bottom])
        mask = np.zeros_like(gt, bool); mask[top:bottom] = True
        region = ("crop", top, 0)
    else:  # DrivingStereo: full frame, resized to width 1216
        arr = read_rgb(s["rgb_path"], size_wh=(W, H))
        h2 = int(round(H * 1216 / W / 8)) * 8
        inp = to_tensor_rgb(np.asarray(Image.fromarray(arr).resize((1216, h2), Image.BILINEAR)))
        mask = np.ones_like(gt, bool)
        region = ("resize", H, W)
    mask &= (gt > 1e-3) & (gt < MAX_D)
    return inp, gt, mask, region


def to_gt_grid(pred, region, gt_shape):
    if region[0] == "resize":
        p = F.interpolate(torch.from_numpy(pred)[None, None], size=gt_shape, mode="bilinear",
                          align_corners=False)[0, 0].numpy()
        return p
    out = np.zeros(gt_shape, np.float32)
    _, top, left = region
    out[top:top + pred.shape[0], left:left + pred.shape[1]] = pred
    return out


def metrics(pred, gt, mask, space="depth"):
    p, g = pred[mask], gt[mask]
    A = np.stack([p, np.ones_like(p)], 1)
    if space == "depth":
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        p = np.clip(s * p + t, 1e-3, MAX_D)
    else:  # disparity-output bases (Lotus): affine fit to inverse depth
        s, t = np.linalg.lstsq(A, 1.0 / g, rcond=None)[0]
        p = np.clip(1.0 / np.clip(s * p + t, 1.0 / MAX_D, 1e3), 1e-3, MAX_D)
    thr = np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "sq_rel": float(np.mean((p - g) ** 2 / g)),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "rmse_log": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "d1": float(np.mean(thr < 1.25)),
        "d2": float(np.mean(thr < 1.25 ** 2)),
        "d3": float(np.mean(thr < 1.25 ** 3)),
    }


class MapHooks:
    """Captures SVDF center-tap weight and IGLEM expert routing on the first UNet call."""

    def __init__(self, model):
        self.data, self.handles = {}, []
        conv = model.unet.conv_in
        if hasattr(conv, "weight_gen"):
            k = conv.ksize
            def svdf_hook(_m, _i, out):
                if "svdf" not in self.data:
                    B, _, h, w = out.shape
                    a = torch.softmax(out.float().view(B, 4, k * k, h, w), dim=2)
                    self.data["svdf_center"] = a[:, :, (k * k) // 2].mean(1)[0].cpu().numpy()
                    ent = -(a * a.clamp_min(1e-8).log()).sum(2).mean(1)
                    self.data["svdf_entropy"] = ent[0].cpu().numpy()
                    self.data["svdf"] = True
            self.handles.append(conv.weight_gen.register_forward_hook(svdf_hook))
        if model.adapter is not None:
            def route_hook(_m, _i, out):
                self.data["routing"] = torch.softmax(out.float(), dim=1)[0].cpu().numpy()
            self.handles.append(model.adapter.residual.router.register_forward_hook(route_hook))

    def pop(self):
        d, self.data = self.data, {}
        d.pop("svdf", None)
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["marigold", "igle", "lora"])
    ap.add_argument("--lora_rank", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="all", help="comma list or 'all'")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save_maps", type=int, default=0, help="save maps for first N images per set")
    ap.add_argument("--save_pred", action="store_true")
    ap.add_argument("--base", default="marigold", help="marigold | e2eft | path")
    ap.add_argument("--single_step", action="store_true")
    ap.add_argument("--proc_w", type=int, default=0, help="inference width (0 = native input)")
    ap.add_argument("--space", default="depth", choices=["depth", "disparity"], help="alignment space")
    ap.add_argument("--pad", type=int, default=0,
                    help="pad multiple (0 = 64 for igle, 8 otherwise); set 64 to pad every mode alike")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthModel(args.mode, lora_rank=args.lora_rank, base=args.base, single_step=args.single_step)
    if args.ckpt:
        ck = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, "revision", args.ckpt)
        model.load_trainable(ck)
    model.to(dev).eval()
    hooks = MapHooks(model) if args.save_maps else None

    files = sorted(glob.glob(f"{MAN}/test_*.json"))
    sets = [os.path.basename(f)[5:-5] for f in files]
    if args.sets != "all":
        sets = [s for s in sets if s in args.sets.split(",")]
    out_dir = f"{ROOT}/revision/eval/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    summary = {"args": vars(args)}
    for set_name in sets:
        samples = json.load(open(f"{MAN}/test_{set_name}.json"))["samples"]
        if args.limit:
            samples = samples[: args.limit]
        rows, times = [], []
        for i, s in enumerate(tqdm(samples, desc=f"{args.tag}:{set_name}")):
            inp, gt, mask, region = prepare(s, set_name)
            H0, W0 = inp.shape[1:]
            x = resize_t(inp[None].to(dev), proc_size(H0, W0, args.proc_w))
            # IGLEM's windowed attention needs latent sides divisible by its window chain
            x, h, w = pad_to_multiple(x, args.pad or (64 if args.mode == "igle" else 8))
            g = torch.Generator(device=dev).manual_seed(args.seed + i)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            pred = model.predict(x, steps=args.steps, generator=g)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - t0)
            pred = resize_t(pred[:, :, :h, :w].float(), (H0, W0))[0, 0].cpu().numpy()
            full = to_gt_grid(pred, region, gt.shape)
            m = metrics(full, gt, mask, args.space)  # eval masks lie inside the processed crop
            rows.append(dict(id=s["sample_id"], **m))
            if hooks is not None and i < args.save_maps:
                np.savez_compressed(f"{out_dir}/maps_{set_name}_{i:04d}.npz", pred=pred,
                                    rgb=((inp.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8),
                                    **hooks.pop())
            if args.save_pred:
                np.save(f"{out_dir}/pred_{set_name}_{i:04d}.npy", pred.astype(np.float16))
        with open(f"{out_dir}/per_image_{set_name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"}
        mean["sec_per_image"] = float(np.median(times[1:] if len(times) > 1 else times))
        mean["n"] = len(rows)
        summary[set_name] = mean
        print(set_name, {k: round(v, 4) for k, v in mean.items()})
    save_json(summary, f"{out_dir}/summary.json")


if __name__ == "__main__":
    main()

