"""Post-hoc analyses for the revision.

  complexity : params / trainable params / GFLOPs per UNet step / latency  (E1)
  stats      : seed mean+-std and paired Wilcoxon + bootstrap CI           (E3)
  figures    : SVDF kernel and IGLEM routing maps, failure cases           (E4, E5)

  python analysis_rev.py complexity
  python analysis_rev.py stats --a igle_v1 --b marigold_v1 --seeds 2026,1,2,3,4
  python analysis_rev.py figures --tag igle_v1_maps --base marigold_v1
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rev_common import KB_H, KB_W, ROOT, DepthModel, count_params, save_json  # noqa: E402

EV = f"{ROOT}/revision/eval"
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "d1", "d2", "d3"]


# ------------------------------------------------------------------ E1
def complexity(args):
    from fvcore.nn import FlopCountAnalysis
    dev = "cuda"
    rows = []
    specs = [("Marigold-v1", "marigold", 0), ("IGLEDepth-v1", "igle", 0)]
    if args.lora_rank:
        specs.append((f"LoRA-r{args.lora_rank}", "lora", args.lora_rank))
    for name, mode, r in specs:
        m = DepthModel(mode, lora_rank=r).to(dev).eval()
        x = torch.randn(1, 3, KB_H, KB_W, device=dev).clamp(-1, 1)
        lat = torch.randn(1, 4, KB_H // 8, KB_W // 8, device=dev)
        t = torch.tensor([999], device=dev)
        with torch.no_grad():
            g = m.guide(x, lat)
            unet_fl = FlopCountAnalysis(_UnetStep(m), (g, lat, t)).unsupported_ops_warnings(False)
            unet_fl = unet_fl.total() / 1e9
            ad_fl = (FlopCountAnalysis(m.adapter, (x, lat)).unsupported_ops_warnings(False).total() / 1e9
                     if m.adapter is not None else 0.0)
            lat_s = {}
            for steps in (1, 4):
                for _ in range(3):
                    m.predict(x, steps=steps)
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(10):
                    m.predict(x, steps=steps)
                torch.cuda.synchronize(); lat_s[steps] = (time.time() - t0) / 10
        extra = {}
        if mode == "igle":
            extra = {"IGLEM_params": count_params(m.adapter),
                     "SVDF_kernel_gen_params": count_params(m.unet.conv_in.weight_gen)}
        rows.append({"model": name, "total_params_M": count_params(m) / 1e6,
                     "trainable_params_M": count_params(m, True) / 1e6,
                     "trainable_pct": 100 * count_params(m, True) / count_params(m),
                     "unet_step_GFLOPs": unet_fl, "adapter_GFLOPs": ad_fl,
                     "latency_1step_s": lat_s[1], "latency_4step_s": lat_s[4],
                     "peak_mem_GB": torch.cuda.max_memory_allocated() / 2**30, **extra})
        print(rows[-1])
        del m; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    save_json({"gpu": torch.cuda.get_device_name(), "input": [KB_H, KB_W], "fp32": True, "rows": rows},
              f"{EV}/complexity.json")


class _UnetStep(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, g, lat, t):
        return self.m.unet_forward(g, lat, t)


# ------------------------------------------------------------------ E3
def read_csv(tag, set_name):
    with open(f"{EV}/{tag}/per_image_{set_name}.csv") as f:
        return list(csv.DictReader(f))


def stats(args):
    from scipy.stats import wilcoxon
    sets = sorted(os.path.basename(p)[10:-4] for p in glob.glob(f"{EV}/{args.a}/per_image_*.csv"))
    seeds = args.seeds.split(",")
    out = {}
    for s in sets:
        res = {}
        for tag in (args.a, args.b):
            vals = []
            for sd in seeds:
                t = tag if sd == seeds[0] else f"{tag}_s{sd}"
                if os.path.exists(f"{EV}/{t}/per_image_{s}.csv"):
                    vals.append(np.mean([float(r["abs_rel"]) for r in read_csv(t, s)]))
            res[tag] = {"abs_rel_mean": float(np.mean(vals)), "abs_rel_std": float(np.std(vals)),
                        "n_seeds": len(vals)}
        a = np.array([float(r["abs_rel"]) for r in read_csv(args.a, s)])
        b = np.array([float(r["abs_rel"]) for r in read_csv(args.b, s)])
        diff = a - b
        rng = np.random.default_rng(0)
        boots = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        res["paired"] = {"n": len(diff), "mean_diff": float(diff.mean()),
                         "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                         "frac_a_better": float((diff < 0).mean()),
                         "wilcoxon_p": float(wilcoxon(a, b).pvalue)}
        out[s] = res
        print(s, json.dumps(res))
    save_json(out, f"{EV}/stats_{args.a}_vs_{args.b}.json")


# ------------------------------------------------------------------ E4 / E5
def _set_of(path):
    # maps_<set>_<idx>.npz
    return os.path.basename(path)[len("maps_"):].rsplit("_", 1)[0]


def figures(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    od = f"{EV}/figures"; os.makedirs(od, exist_ok=True)
    maps = sorted(glob.glob(f"{EV}/{args.tag}/maps_*.npz"))
    route_stats = {}
    for p in maps:
        d = np.load(p)
        set_name = _set_of(p)
        if "routing" in d:
            r = d["routing"]  # [N, h, w]
            route_stats.setdefault(set_name, []).append(r.reshape(r.shape[0], -1).mean(1))
    summary = {k: {"mean_weight_per_expert": np.mean(v, 0).tolist(),
                   "std_over_images": np.std(v, 0).tolist()} for k, v in route_stats.items()}
    save_json(summary, f"{od}/routing_summary.json")
    # one panel per weather set: rgb | prediction | SVDF center tap | dominant expert
    picks = {}
    for p in maps:
        picks.setdefault(_set_of(p), p)
    fig, axes = plt.subplots(len(picks), 4, figsize=(16, 2.2 * len(picks)))
    axes = np.atleast_2d(axes)
    for row, (set_name, p) in zip(axes, sorted(picks.items())):
        d = np.load(p)
        row[0].imshow(d["rgb"]); row[0].set_ylabel(set_name, fontsize=8)
        row[1].imshow(d["pred"], cmap="Spectral")
        if "svdf_center" in d:
            im = row[2].imshow(d["svdf_center"], cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, ax=row[2], fraction=0.02)
        if "routing" in d:
            row[3].imshow(d["routing"].argmax(0), cmap="tab10", vmin=0, vmax=9)
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    for a, t in zip(axes[0], ["input", "prediction", "SVDF center-tap weight", "IGLEM dominant expert"]):
        a.set_title(t, fontsize=9)
    fig.tight_layout(); fig.savefig(f"{od}/mechanism.pdf"); fig.savefig(f"{od}/mechanism.png", dpi=150)
    # failure cases: worst AbsRel of the reference model on the real-weather sets
    worst = {}
    for s in ("cadc_snow", "ds_rain", "ds_fog"):
        f = f"{EV}/{args.base_tag}/per_image_{s}.csv"
        if os.path.exists(f):
            rows = sorted(read_csv(args.base_tag, s), key=lambda r: -float(r["abs_rel"]))[:5]
            worst[s] = [(r["id"], float(r["abs_rel"])) for r in rows]
    save_json(worst, f"{od}/failure_cases.json")
    print("figures written to", od)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["complexity", "stats", "figures"])
    ap.add_argument("--a", default="igle_v1")
    ap.add_argument("--b", default="marigold_v1")
    ap.add_argument("--seeds", default="2026,1,2,3,4")
    ap.add_argument("--tag", default="igle_v1_maps")
    ap.add_argument("--base_tag", default="igle_v1")
    ap.add_argument("--lora_rank", type=int, default=0)
    a = ap.parse_args()
    {"complexity": complexity, "stats": stats, "figures": figures}[a.what](a)
