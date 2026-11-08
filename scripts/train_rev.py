"""Input-path training of a frozen diffusion depth base (Marigold, E2E-FT or Lotus-G).

--mode marigold trains only the UNet input convolution (conv_in control, 23,360 params);
--mode igle trains IGLEM + SVDF + conv_in (IGLEDepth, 78,315 params).
--target teacher: clear-weather self-distillation, the frozen base's own prediction on the
clear frame (from gen_teacher.py) supervises every weather rendering of that frame.
--target dav2: dense Depth Anything V2 pseudo-labels (gen_pseudo.py), the earlier setting.

Schedule: --warmup_iters on clear frames, then --iters on clear + weather renderings,
effective batch 32 (micro-batch 4), Adam 1e-4 with 200-iteration warm-up and exponential
decay to 1%, bf16 autocast, horizontal flips. Multi-step bases use the v-prediction loss with
multi-resolution noise; --single_step bases use an MSE on the clean-latent estimate.

Resumable: re-running with the same --out continues from out/last.pt.

Examples (paper settings):
  python train_rev.py --mode igle --target teacher --out runs/igledepth_sd --warmup_iters 1000 --iters 9000
  python train_rev.py --mode igle --base e2eft --single_step --proc_w 768 --target teacher \
      --teacher_dir teacher_e2eft --out runs/igle_e2eft_sd --warmup_iters 1000 --iters 9000
  python train_rev.py --mode igle --out runs/igledepth_v1          # DAv2 pseudo-labels, 2k + 12k
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import (KB_H, KB_W, ROOT, DepthModel, count_params, kb_crop_box,  # noqa: E402
                        norm_depth, proc_size, read_rgb, resize_t, save_json, to_tensor_rgb)
from src.util.multi_res_noise import multi_res_noise_like  # noqa: E402


class TrainSet(Dataset):
    def __init__(self, frames, weather, flip_p=0.5, target="dav2", teacher_dir="teacher_mg", proc_w=0):
        self.target, self.teacher_dir = target, teacher_dir
        self.hw = proc_size(KB_H, KB_W, proc_w)
        self.items = []
        for i, fr in enumerate(frames):
            self.items.append((i, None))
            if weather:
                for dom in fr["weather"]:
                    self.items.append((i, dom))
        self.frames, self.flip_p = frames, flip_p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, dom = self.items[k]
        fr = self.frames[i]
        clear = Image.open(fr["rgb"])
        native = clear.size  # (W, H)
        arr = read_rgb(fr["weather"][dom] if dom else fr["rgb"], size_wh=native)
        top, left = kb_crop_box(native[1], native[0])
        arr = arr[top:top + KB_H, left:left + KB_W]
        rgb = to_tensor_rgb(arr)
        if self.target == "teacher":  # clear-image prediction of the frozen base, already in [0,1]
            path = fr["pseudo"].replace("pseudo_dav2", self.teacher_dir)
            depth = torch.from_numpy(np.asarray(Image.open(path), dtype=np.float32) / 65535.0)[None] * 2 - 1
        else:
            depth = torch.from_numpy(np.asarray(Image.open(fr["pseudo"]), dtype=np.float32) / 256.0)[None]
            depth = norm_depth(depth, depth > 1e-3)
        rgb, depth = resize_t(rgb, self.hw), resize_t(depth, self.hw)
        if random.random() < self.flip_p:
            rgb, depth = rgb.flip(-1), depth.flip(-1)
        return rgb, depth


def lr_lambda(warmup, total, final_ratio=0.01):
    def f(it):
        if it < warmup:
            return (it + 1) / warmup
        return final_ratio ** ((it - warmup) / max(1, total - warmup))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["marigold", "igle", "lora"])
    ap.add_argument("--lora_rank", type=int, default=0)
    ap.add_argument("--target", default="dav2", choices=["dav2", "teacher"],
                    help="dav2: LiDAR-aligned DAv2 pseudo-labels; teacher: clear-image self-distillation")
    ap.add_argument("--teacher_dir", default="teacher_mg")
    ap.add_argument("--base", default="marigold", help="marigold | e2eft | path")
    ap.add_argument("--single_step", action="store_true", help="E2E-FT-style deterministic base")
    ap.add_argument("--proc_w", type=int, default=0, help="train at this width (0 = native crop)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup_iters", type=int, default=2000, help="clear-only stage")
    ap.add_argument("--iters", type=int, default=12000, help="weather stage")
    ap.add_argument("--batch", type=int, default=4, help="micro batch per step")
    ap.add_argument("--eff_batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--max_minutes", type=float, default=0, help="stop cleanly after N minutes")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, "revision", args.out)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    dev = "cuda"

    frames = json.load(open(f"{ROOT}/revision/manifests/train.json"))["frames"]
    frames = [f for f in frames if os.path.exists(f["pseudo"])]
    if args.target == "teacher":
        frames = [f for f in frames if os.path.exists(f["pseudo"].replace("pseudo_dav2", args.teacher_dir))]
    if args.subset < 1:
        rng = random.Random(0)
        frames = rng.sample(frames, int(len(frames) * args.subset))
    doms = sorted({d for f in frames for d in f["weather"]})
    print(f"frames {len(frames)}, weather domains {doms}")

    model = DepthModel(args.mode, lora_rank=args.lora_rank, base=args.base,
                       single_step=args.single_step).to(dev)
    if args.grad_ckpt:
        model.unet.enable_gradient_checkpointing()
    model.vae.to(torch.bfloat16)
    params = [p for p in model.parameters() if p.requires_grad]
    info = {"mode": args.mode, "trainable_params": sum(p.numel() for p in params),
            "total_params": count_params(model), "frames": len(frames), "weather_domains": doms,
            "args": vars(args)}
    save_json(info, f"{out}/info.json")
    print(info)

    opt = torch.optim.Adam(params, lr=args.lr)
    total = args.warmup_iters + args.iters
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(200, total))
    accum = args.eff_batch // args.batch
    it = 0
    if os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); it = ck["iter"]
        print(f"resumed at iter {it}")

    def loader(weather):
        ds = TrainSet(frames, weather, target=args.target, teacher_dir=args.teacher_dir, proc_w=args.proc_w)
        g = torch.Generator(); g.manual_seed(args.seed + it)
        return DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                          drop_last=True, generator=g, persistent_workers=True, pin_memory=True)

    ns = model.train_sched
    T = ns.config.num_train_timesteps
    t0, log = time.time(), open(f"{out}/train_log.txt", "a")
    stage = None
    while it < total:
        want = "weather" if it >= args.warmup_iters else "clear"
        if stage != want:
            stage, dl = want, iter(loader(want == "weather"))
        loss_acc = 0.0
        for _ in range(accum):
            try:
                rgb, depth = next(dl)
            except StopIteration:
                dl = iter(loader(stage == "weather")); rgb, depth = next(dl)
            rgb, depth = rgb.to(dev, non_blocking=True), depth.to(dev, non_blocking=True)
            with torch.no_grad():
                rgb_lat = model.encode(rgb.bfloat16()).float()
                tgt = model.encode(depth.repeat(1, 3, 1, 1).bfloat16()).float()
            if args.single_step:  # deterministic t=T step, loss on the clean-latent estimate
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model.single_step_x0(model.guide(rgb, rgb_lat))
                loss = torch.nn.functional.mse_loss(pred.float(), tgt) / accum
            else:
                t = torch.randint(0, T, (rgb.shape[0],), device=dev)
                noise = multi_res_noise_like(tgt, strength=0.9 * (t / T), downscale_strategy="original",
                                             device=dev)
                noisy = ns.add_noise(tgt, noise, t)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    guide = model.guide(rgb, rgb_lat)
                    pred = model.unet_forward(guide, noisy, t)
                target = ns.get_velocity(tgt, noise, t)
                loss = torch.nn.functional.mse_loss(pred.float(), target) / accum
            loss.backward()
            loss_acc += loss.item()
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        it += 1
        if it % 20 == 0:
            el = time.time() - t0
            msg = f"iter {it}/{total} stage {stage} loss {loss_acc:.5f} lr {sched.get_last_lr()[0]:.2e} {el/60:.1f} min, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        stop = args.max_minutes and (time.time() - t0) / 60 > args.max_minutes
        if it % args.save_every == 0 or it == total or stop:
            torch.save({"model": model.trainable_state(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "iter": it}, f"{out}/last.pt.tmp")
            os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")
        if stop:
            print("time limit reached; resume by re-running the same command"); return
    torch.save(model.trainable_state(), f"{out}/final_trainable.pt")
    print("done", out)


if __name__ == "__main__":
    main()

