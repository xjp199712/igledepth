"""Paper extras for the IEEE Access version: clean latency/memory/params benchmark of every
plug-in configuration, qualitative predictions on representative images, and IGLEM/SVDF
maps on Marigold. Writes figs/bench.json and figs/qual_<base>_<set>.npz; plotting is in
figs_plot.py.

Representative images: for each base and set, the image at the 10th percentile of the
per-image AbsRel change (IGLEM+SVDF minus zero-shot), i.e. a clear but not extreme gain.
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_rev import MAN, MAX_D, MapHooks, prepare, to_gt_grid  # noqa: E402
from rev_common import ROOT, DepthModel, count_params, pad_to_multiple, proc_size, resize_t  # noqa: E402

OUT = f"{ROOT}/revision/figs"
EVAL = f"{ROOT}/revision/eval"
# name: (base, mode, ckpt, single_step, proc_w, pad, space)
CFG = {
    "marigold_zs": ("marigold", "marigold", None, False, 0, 8, "depth"),
    "marigold_sd": ("marigold", "marigold", "runs/marigold_sd/final_trainable.pt", False, 0, 8, "depth"),
    "igle_sd": ("marigold", "igle", "runs/igledepth_sd/final_trainable.pt", False, 0, 64, "depth"),
    "e2eft_zs": ("e2eft", "marigold", None, True, 768, 8, "depth"),
    "convin_e2eft_sd": ("e2eft", "marigold", "runs/convin_e2eft_sd/final_trainable.pt", True, 768, 8, "depth"),
    "igle_e2eft_sd": ("e2eft", "igle", "runs/igle_e2eft_sd/final_trainable.pt", True, 768, 64, "depth"),
    "lotusg_zs": ("lotus_g", "marigold", None, True, 768, 64, "disparity"),
    "convin_lotusg_sd": ("lotus_g", "marigold", "runs/convin_lotusg_sd/final_trainable.pt", True, 768, 64, "disparity"),
    "igle_lotusg_sd": ("lotus_g", "igle", "runs/igle_lotusg_sd/final_trainable.pt", True, 768, 64, "disparity"),
}
GROUPS = {"marigold": ["marigold_zs", "marigold_sd", "igle_sd"],
          "e2eft": ["e2eft_zs", "convin_e2eft_sd", "igle_e2eft_sd"],
          "lotus_g": ["lotusg_zs", "convin_lotusg_sd", "igle_lotusg_sd"]}
QUAL_SETS = ["ds_rain", "ds_fog", "cadc_snow", "kitti_mix_snow"]
N_BENCH = 30


def per_image(tag, set_name):
    with open(f"{EVAL}/{tag}/per_image_{set_name}.csv") as f:
        return np.array([float(r["abs_rel"]) for r in csv.DictReader(f)])


def pick(base, set_name):
    zs, ours = GROUPS[base][0], GROUPS[base][2]
    d = per_image(ours, set_name) - per_image(zs, set_name)
    return int(np.argsort(d)[int(0.1 * len(d))])


def aligned(full, gt, mask, space):
    p, g = full[mask], gt[mask]
    A = np.stack([p, np.ones_like(p)], 1)
    if space == "depth":
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        return np.clip(s * full + t, 1e-3, MAX_D)
    s, t = np.linalg.lstsq(A, 1.0 / g, rcond=None)[0]
    return np.clip(1.0 / np.clip(s * full + t, 1.0 / MAX_D, 1e3), 1e-3, MAX_D)


def run(model, inp, cfg, gen_seed):
    _, _, _, _, proc_w, pad, _ = cfg
    H0, W0 = inp.shape[1:]
    x = resize_t(inp[None].cuda(), proc_size(H0, W0, proc_w))
    x, h, w = pad_to_multiple(x, pad)
    g = torch.Generator(device="cuda").manual_seed(gen_seed)
    pred = model.predict(x, steps=4, generator=g)
    return resize_t(pred[:, :, :h, :w].float(), (H0, W0))[0, 0].cpu().numpy(), (h, w)


def main():
    os.makedirs(OUT, exist_ok=True)
    picks = {(b, s): pick(b, s) for b in GROUPS for s in QUAL_SETS}
    samples = {s: json.load(open(f"{MAN}/test_{s}.json"))["samples"] for s in QUAL_SETS + ["kitti_sunny"]}
    bench, qual = {}, {}
    for name, cfg in CFG.items():
        base, mode, ckpt, single, *_ = cfg
        model = DepthModel(mode, base=base, single_step=single)
        if ckpt:
            model.load_trainable(f"{ROOT}/revision/{ckpt}")
        model.cuda().eval()
        hooks = MapHooks(model) if mode == "igle" and base == "marigold" else None
        # ---- benchmark on KITTI sunny (352x1216 input)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for i, s in enumerate(samples["kitti_sunny"][: N_BENCH + 3]):
            inp, *_ = prepare(s, "kitti_sunny")
            torch.cuda.synchronize(); t0 = time.time()
            run(model, inp, cfg, 2026 + i)
            torch.cuda.synchronize()
            if i >= 3:
                times.append(time.time() - t0)
        n_in = count_params(model.trainable_modules())  # conv_in (+ IGLEM/SVDF)
        bench[name] = {"base": base, "mode": mode,
                       "trainable_params": 0 if ckpt is None else n_in,
                       "total_params": count_params(model),
                       "input_module_params": n_in,
                       "latency_ms_median": 1000 * float(np.median(times)),
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        print(name, bench[name], flush=True)
        # ---- qualitative
        for set_name in QUAL_SETS:
            idx = picks[(base, set_name)]
            s = samples[set_name][idx]
            inp, gt, mask, region = prepare(s, set_name)
            pred, (h, w) = run(model, inp, cfg, 2026 + idx)  # same noise seed as eval_rev
            full = to_gt_grid(pred, region, gt.shape)
            dep = aligned(full, gt, mask, cfg[6])
            absrel = float(np.mean(np.abs(dep[mask] - gt[mask]) / gt[mask]))
            key = (base, set_name)
            q = qual.setdefault(key, {"idx": idx, "id": s["sample_id"], "gt": gt, "mask": mask,
                                     "region": np.array([region[0] == "crop", *region[1:]], dtype=object),
                                     "rgb": ((inp.permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)})
            q[f"pred_{name}"] = dep.astype(np.float32)
            q[f"absrel_{name}"] = absrel
            if hooks is not None:
                m = hooks.pop()
                lh, lw = h // 8, w // 8
                for k in ("svdf_center", "svdf_entropy"):
                    if k in m:
                        q[k] = m[k][:lh, :lw]
                if "routing" in m:
                    q["routing"] = m["routing"][:, :lh, :lw]
        del model
        torch.cuda.empty_cache()
    json.dump(bench, open(f"{OUT}/bench.json", "w"), indent=1)
    for (base, set_name), q in qual.items():
        np.savez_compressed(f"{OUT}/qual_{base}_{set_name}.npz", **{k: v for k, v in q.items() if k != "region"})
        print("qual", base, set_name, q["id"], {k: round(v, 4) for k, v in q.items() if k.startswith("absrel")})
    print("FIGS_RUN_DONE")


if __name__ == "__main__":
    main()

