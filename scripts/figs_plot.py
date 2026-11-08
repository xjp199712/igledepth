"""Plots for the IEEE Access version from figs_run.py outputs and eval summaries:
qual_<base>.png/pdf, mech_marigold.png/pdf, overview.png/pdf, bench table (bench.tex)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.environ.get("IGLE_ROOT", "/workspace/igledepth_v2")  # data, checkpoints and outputs
OUT = f"{ROOT}/revision/figs"
EVAL = f"{ROOT}/revision/eval"
SETS = ["ds_rain", "ds_fog", "cadc_snow", "kitti_mix_snow"]
SET_LABEL = {"ds_rain": "DS\nrain", "ds_fog": "DS\nfog", "cadc_snow": "CADC\nsnow", "kitti_mix_snow": "KITTI\nmix-snow"}
WEATHER = ["kitti_raingan", "kitti_snowgan", "kitti_mix_snow", "kitti_mix_rain", "kitti_fog75",
           "kitti_fog150", "ds_rain", "ds_fog", "cadc_snow"]
GROUPS = {"marigold": ["marigold_zs", "marigold_sd", "igle_sd"],
          "e2eft": ["e2eft_zs", "convin_e2eft_sd", "igle_e2eft_sd"],
          "lotus_g": ["lotusg_zs", "convin_lotusg_sd", "igle_lotusg_sd"]}
BASE_LABEL = {"marigold": "Marigold", "e2eft": "E2E-FT", "lotus_g": "Lotus-G"}
COL_LABEL = ["Zero-shot", "Self-distill. (conv_in)", "Self-distill. (IGLEM+SVDF)"]
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})


def dilate(a, r=2):
    """Grow sparse LiDAR points (NaN elsewhere) so they are visible at print size."""
    out = a.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            sh = np.roll(np.roll(a, dy, 0), dx, 1)
            fill = np.isnan(out) & ~np.isnan(sh)
            out[fill] = sh[fill]
    return out


def ratios(base):
    out = []
    for s in SETS:
        q = np.load(f"{OUT}/qual_{base}_{s}.npz", allow_pickle=True)
        rs, cs = bbox(q["mask"])
        out.append((rs.stop - rs.start) / (cs.stop - cs.start))
    return out


def bbox(mask):
    r, c = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return slice(r[0], r[-1] + 1), slice(c[0], c[-1] + 1)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def qual(base):
    hr = ratios(base)
    fig, ax = plt.subplots(len(SETS), 5, figsize=(12, 2.45 * sum(hr) + 0.6),
                           gridspec_kw={"height_ratios": hr, "hspace": 0.28, "wspace": 0.03})
    for r, s in enumerate(SETS):
        q = np.load(f"{OUT}/qual_{base}_{s}.npz", allow_pickle=True)
        mask, gt = q["mask"], q["gt"]
        rs, cs = bbox(mask)
        inv = 1.0 / gt[mask]
        lo, hi = np.percentile(inv, 2), np.percentile(inv, 98)
        ax[r, 0].imshow(q["rgb"], aspect="auto")
        ax[r, 0].set_ylabel(SET_LABEL[s])
        g = np.where(mask, 1.0 / np.maximum(gt, 1e-3), np.nan)[rs, cs]
        ax[r, 1].set_facecolor("black")
        ax[r, 1].imshow(dilate(g), cmap="turbo", vmin=lo, vmax=hi, interpolation="nearest", aspect="auto")
        for c, name in enumerate(GROUPS[base]):
            ax[r, 2 + c].imshow((1.0 / q[f"pred_{name}"])[rs, cs], cmap="turbo", vmin=lo, vmax=hi, aspect="auto")
            ax[r, 2 + c].set_title(f"AbsRel {float(q[f'absrel_{name}']):.3f}", fontsize=7, pad=2)
        if r == 0:
            for c, t in enumerate(["Input", "Ground truth"] + COL_LABEL):
                ax[r, c].annotate(t, (0.5, 1.22), xycoords="axes fraction", ha="center", fontsize=8,
                                  fontweight="bold")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    save(fig, f"qual_{base}")


def mech():
    hr = ratios("marigold")
    fig, ax = plt.subplots(len(SETS), 4, figsize=(10, 2.45 * sum(hr) * 1.15 + 0.4),
                           gridspec_kw={"height_ratios": hr, "hspace": 0.12, "wspace": 0.32})
    for r, s in enumerate(SETS):
        q = np.load(f"{OUT}/qual_marigold_{s}.npz", allow_pickle=True)
        H, W = q["rgb"].shape[:2]
        ext = (0, W, H, 0)
        ax[r, 0].imshow(q["rgb"], aspect="auto"); ax[r, 0].set_ylabel(SET_LABEL[s])
        rt = q["routing"].transpose(1, 2, 0)
        lo_, hi_ = np.percentile(rt, 1, axis=(0, 1)), np.percentile(rt, 99, axis=(0, 1))
        ax[r, 1].imshow(np.clip((rt - lo_) / (hi_ - lo_ + 1e-6), 0, 1), extent=ext, interpolation="bilinear",
                        aspect="auto")
        im = ax[r, 2].imshow(q["svdf_center"], cmap="viridis", extent=ext, aspect="auto")
        fig.colorbar(im, ax=ax[r, 2], fraction=0.025, pad=0.01)
        im = ax[r, 3].imshow(q["svdf_entropy"], cmap="magma", extent=ext, aspect="auto")
        fig.colorbar(im, ax=ax[r, 3], fraction=0.025, pad=0.01)
        if r == 0:
            for c, t in enumerate(["Input", "IGLEM routing (RGB = 3 experts)",
                                   "SVDF centre-tap weight", "SVDF kernel entropy"]):
                ax[r, c].set_title(t, fontsize=8, fontweight="bold")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    save(fig, "mech_marigold")


def overview():
    def stats(tag):
        d = json.load(open(f"{EVAL}/{tag}/summary.json"))
        return d["kitti_sunny"]["abs_rel"], float(np.mean([d[s]["abs_rel"] for s in WEATHER]))
    ext = {"ext_weatherdepth": "WeatherDepth*", "ext_robustdepth": "Robust-Depth*", "ext_dav2": "DAv2-L",
           "ext_dav2s": "DAv2-S", "ext_daw": "DA-W", "ext_daac": "DA-AC", "ext_lotus_d": "Lotus-D",
           "ext_depthmaster": "DepthMaster", "ext_da3large": "DA3-L", "ext_depthpro": "Depth Pro",
           "ext_moge2": "MoGe-2", "ext_da3mono": "DA3-Mono"}
    fig, a = plt.subplots(figsize=(6.2, 4.6))
    xs = np.linspace(0.10, 0.18, 2)
    for k in (1.0, 1.1, 1.2, 1.3):
        a.plot(xs, k * xs, ls=":", lw=0.8, color="0.6")
        xe = min(0.18, 0.20 / k) - 0.002
        a.text(xe, k * xe, f"ratio {k:.1f}", fontsize=6.5, color="0.45", ha="right", va="bottom",
               rotation=33)
    for tag, lab in ext.items():
        x, y = stats(tag)
        if y > 0.2:
            a.annotate(f"{lab} ({x:.3f}, {y:.3f})", (x, 0.197), fontsize=6.5, color="0.4", ha="center",
                       arrowprops=dict(arrowstyle="->", color="0.6", lw=0.6), xytext=(x, 0.188))
            continue
        a.scatter(x, y, s=14, color="0.55", zorder=2)
        if lab == "DepthMaster":  # crowded area: label in open space with a leader line
            a.annotate(lab, (x, y), xytext=(0.147, 0.169), fontsize=6.5, color="0.35",
                       arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
            continue
        off = {"DA-AC": (4, -3), "DAv2-S": (-30, 4), "DA-W": (3, -9), "Robust-Depth*": (-8, 6), "DAv2-L": (4, -4),
               "WeatherDepth*": (4, -6), "DepthMaster": (-22, -11)}.get(lab, (3, 2))
        a.annotate(lab, (x, y), fontsize=6.5, color="0.35", xytext=off, textcoords="offset points")
    colors = {"marigold": "#1f77b4", "e2eft": "#2ca02c", "lotus_g": "#d62728"}
    for base, (zs, ctl, ours) in GROUPS.items():
        (x0, y0), (x1, y1), (x2, y2) = stats(zs), stats(ctl), stats(ours)
        col = colors[base]
        a.scatter(x0, y0, s=26, facecolors="none", edgecolors=col, zorder=3)
        a.scatter(x1, y1, s=22, marker="s", facecolors="none", edgecolors=col, zorder=3)
        a.scatter(x2, y2, s=34, color=col, zorder=4, label=BASE_LABEL[base])
        a.annotate("", (x2, y2), (x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    a.scatter([], [], s=26, facecolors="none", edgecolors="k", label="zero-shot")
    a.scatter([], [], s=22, marker="s", facecolors="none", edgecolors="k", label="self-distill. (conv_in)")
    a.scatter([], [], s=34, color="k", label="self-distill. (IGLEM+SVDF)")
    a.set_xlim(0.10, 0.18); a.set_ylim(0.115, 0.20)
    a.set_xlabel("Clear-weather AbsRel (KITTI sunny)")
    a.set_ylabel("Adverse-weather AbsRel (mean of 9 sets)")
    a.legend(fontsize=6.5, loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.04))
    a.text(0.995, 0.005, "* trained on KITTI (in-domain)", transform=a.transAxes, fontsize=6, ha="right",
           color="0.4")
    save(fig, "overview")


def bench_table():
    b = json.load(open(f"{OUT}/bench.json"))
    rows = []
    for base, names in GROUPS.items():
        for n, lab in zip(names, ["zero-shot", "+ SD (conv\\_in)", "+ SD (IGLEM+SVDF)"]):
            r = b[n]
            rows.append(f"{BASE_LABEL[base]} & {lab} & {r['trainable_params']:,} & "
                        f"{r['total_params'] / 1e6:.1f} & {r['latency_ms_median']:.0f} & {r['peak_mem_gb']:.2f} \\\\")
    open(f"{OUT}/bench.tex", "w").write("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    for base in GROUPS:
        qual(base)
    mech()
    overview()
    bench_table()
    print("FIGS_PLOT_DONE")


