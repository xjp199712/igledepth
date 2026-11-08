# Clear-Weather Self-Distillation (IEEE Access version)

Code for *Label-Free Clear-Weather Self-Distillation for Weather-Robust Diffusion-Based
Monocular Depth Estimation*. A frozen diffusion depth estimator (Marigold v1-1, E2E-FT or
Lotus-G) predicts depth on a clear KITTI frame; the same frozen network, with only its input
path trainable, learns to reproduce that prediction from synthetic rain and snow renderings
of the frame. No depth labels are used for training.

An earlier version of this work, trained with Depth Anything V2 pseudo-labels, is at
[jungletada/lrdm](https://github.com/jungletada/lrdm); this repository supersedes it.

## Repository layout

```
marigold/ramit_model/ramit.py         # IGLEM (RAMiT-based image-guided latent enhancement)
marigold/ramit_model/depth_filter.py  # SVDF (spatially varying depth filter)
src/util/multi_res_noise.py           # multi-resolution noise for the diffusion loss
data_split/kitti_depth/               # Eigen training split
scripts/                              # training, evaluation and figure scripts (below)
```

The `marigold/` modules are taken unchanged from the earlier repository, which is a fork of
[Marigold](https://github.com/prs-eth/Marigold) (Apache-2.0, see `LICENSE`); IGLEM builds on
[RAMiT](https://github.com/rami0205/RAMiT).

## Files in `scripts/`

| File | Purpose |
|---|---|
| `rev_common.py` | Model wrapper: frozen base + trainable input path (`marigold` = conv_in only, `igle` = IGLEM + SVDF + conv_in), single-step and multi-step inference, data helpers |
| `build_manifests.py` | Builds `manifests/train.json` from the WeatherKITTI training split |
| `gen_teacher.py` | Clear-weather teachers of the frozen base (uint16 PNG) |
| `gen_pseudo.py` | Depth Anything V2 pseudo-labels (only for the supervision-source comparison) |
| `train_rev.py` | Self-distillation (`--target teacher`) or pseudo-label training (`--target dav2`) |
| `eval_rev.py` | Evaluation of our configurations on the ten test sets, per-image metrics |
| `eval_ext.py` | Zero-shot evaluation of the 14 external models under the same protocol |
| `analysis_rev.py` | Paired Wilcoxon tests and bootstrap confidence intervals |
| `figs_run.py`, `figs_plot.py` | Latency / memory benchmark, qualitative and mechanism figures, overview plot |
| `run_paper.sh` | Every command used for the paper, in order |
| `manifests/test_*.json` | The exact test samples and ground-truth files of the ten test sets |

## Environment

Our models (`env-v1`): Python 3.10, torch 2.4.1 (CUDA 12.4), torchvision 0.19.1,
diffusers 0.30.3, transformers 4.44.2, accelerate 0.34.2, timm 1.0, einops, numpy, scipy,
pillow, matplotlib.

External models (`env-ext`, used only by `eval_ext.py`): torch 2.5.1, diffusers 0.32.2,
transformers 4.50.3, numpy 1.26, plus the official code of
[Lotus](https://github.com/EnVision-Research/Lotus),
[DepthMaster](https://github.com/indu1ge/DepthMaster),
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3),
[MoGe](https://github.com/microsoft/MoGe),
[Depth Anything AC](https://github.com/HVision-NKU/DepthAnythingAC),
[DA-W](https://github.com/taco-group/DA-W),
[Robust-Depth](https://github.com/kieran514/Robust-Depth) and
[WeatherDepth](https://github.com/wangjiyuan9/WeatherDepth), cloned into
`$IGLE_ROOT/repos/ext/<name>`, with their released checkpoints under
`$IGLE_ROOT/assets/weather_ckpt/`.

## Data layout

All paths are relative to `IGLE_ROOT` (default `/workspace/igledepth_v2`; set the environment
variable to change it).

```
$IGLE_ROOT/
  assets/checkpoints/marigold-depth-v1-1/           # prs-eth/marigold-depth-v1-1
  assets/checkpoints/Depth-Anything-V2-Large-hf/    # only for gen_pseudo.py / eval_ext.py
  data/kitti-full-paired/{rgb,depth}/...            # KITTI raw RGB + projected LiDAR (Eigen train)
  data/weather-full/{raingan,snowgan,mix_snow,mix_rain,fog}/...   # WeatherKITTI renderings
  data/kitti-test/, data/kitti-author-gt/           # 697 Eigen test frames and their LiDAR ground truth
  data/real-weather/{drivingstereo,cadcd}/...       # DrivingStereo rain/fog, CADC snow
  revision/{manifests,teacher_*,runs,eval,figs}     # created by the scripts
```

E2E-FT (`GonzaloMG/marigold-e2e-ft-depth`) and Lotus-G
(`jingheya/lotus-depth-g-v2-1-disparity`) are loaded from the Hugging Face Hub.

Training uses only the `raingan`, `snowgan` and `mix_snow` renderings; fog, mixed rain and
the real-weather sets are unseen. The shipped test manifests contain absolute paths under
`/workspace/igledepth_v2`; copy `scripts/manifests/test_*.json` to `$IGLE_ROOT/revision/manifests/`
and, if your data lives elsewhere, copy them to
`$IGLE_ROOT/revision/manifests/` and rewrite the prefix, e.g.

```
sed -i "s#/workspace/igledepth_v2#$IGLE_ROOT#g" $IGLE_ROOT/revision/manifests/test_*.json
```

## Reproducing the paper

`bash scripts/run_paper.sh` runs everything in order. Where each result comes from:

| Paper item | Produced by |
|---|---|
| Main results on three bases (zero-shot / conv_in / IGLEDepth) | `train_rev.py` + `eval_rev.py`, tags `*_zs`, `*_sd` |
| Paired comparison counts and p-values | `analysis_rev.py stats` |
| External model benchmark (14 models) | `eval_ext.py`, tags `ext_*` |
| Latency, memory and parameters | `figs_run.py` (`figs/bench.json`) and `figs_plot.py` (`bench.tex`) |
| Supervision source (pseudo-labels vs. self-distillation) | `gen_pseudo.py`, runs `marigold_v1`, `igledepth_v1` |
| Overview, qualitative and mechanism figures | `figs_run.py` + `figs_plot.py` |

Protocol: per-image least-squares scale and shift alignment (depth space, or disparity space
for disparity-output models), 80 m cap, Eigen crop on KITTI, rows 232-776 on CADC, full frame
on DrivingStereo. Every configuration is trained once with seed 2025; evaluation uses seed 2026.
