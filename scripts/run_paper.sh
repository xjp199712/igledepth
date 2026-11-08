#!/usr/bin/env bash
# Reproduce every number, table and figure of the IEEE Access paper
# "Label-Free Clear-Weather Self-Distillation for Weather-Robust Diffusion-Based
# Monocular Depth Estimation". Can be started from any directory.
# Each training run takes about 3.5 h on one RTX 4090; each evaluation about 30-60 min.
set -e
cd "$(dirname "$0")"
export IGLE_ROOT=${IGLE_ROOT:-/workspace/igledepth_v2}

# 0. Training manifest (test manifests are shipped in manifests/; see README).
python build_manifests.py

# 1. Clear-weather teachers (frozen base on the clear frame).
python gen_teacher.py --steps 1 --batch 8
python gen_teacher.py --base e2eft   --single_step --proc_w 768 --out_dir teacher_e2eft  --batch 8
python gen_teacher.py --base lotus_g --single_step --proc_w 768 --out_dir teacher_lotusg --batch 8

# 2. Self-distillation: conv_in control (--mode marigold) and IGLEDepth (--mode igle), 1k + 9k iterations.
SD="--target teacher --warmup_iters 1000 --iters 9000 --batch 4"
python train_rev.py --mode marigold $SD --out runs/marigold_sd
python train_rev.py --mode igle     $SD --out runs/igledepth_sd
E2E="--base e2eft --single_step --proc_w 768 --teacher_dir teacher_e2eft"
python train_rev.py --mode marigold $E2E $SD --out runs/convin_e2eft_sd
python train_rev.py --mode igle     $E2E $SD --out runs/igle_e2eft_sd
LOT="--base lotus_g --single_step --proc_w 768 --teacher_dir teacher_lotusg"
python train_rev.py --mode marigold $LOT $SD --out runs/convin_lotusg_sd
python train_rev.py --mode igle     $LOT $SD --out runs/igle_lotusg_sd

# 3. Earlier supervision (Table "Supervision source"): DAv2 pseudo-labels, 2k + 12k iterations.
python gen_pseudo.py
python train_rev.py --mode marigold --out runs/marigold_v1
python train_rev.py --mode igle     --out runs/igledepth_v1

# 4. Evaluation on the ten test sets (per-image CSVs + summary.json in $IGLE_ROOT/revision/eval/<tag>).
python eval_rev.py --mode marigold --tag marigold_zs
python eval_rev.py --mode marigold --ckpt runs/marigold_sd/final_trainable.pt   --tag marigold_sd
python eval_rev.py --mode igle     --ckpt runs/igledepth_sd/final_trainable.pt  --tag igle_sd
python eval_rev.py --mode marigold --ckpt runs/marigold_v1/final_trainable.pt   --tag marigold_v1
python eval_rev.py --mode igle     --ckpt runs/igledepth_v1/final_trainable.pt  --tag igle_v1
E2E_EV="--base e2eft --single_step --proc_w 768"
python eval_rev.py --mode marigold $E2E_EV --tag e2eft_zs
python eval_rev.py --mode marigold $E2E_EV --ckpt runs/convin_e2eft_sd/final_trainable.pt --tag convin_e2eft_sd
python eval_rev.py --mode igle     $E2E_EV --ckpt runs/igle_e2eft_sd/final_trainable.pt   --tag igle_e2eft_sd
LOT_EV="--base lotus_g --single_step --proc_w 768 --space disparity --pad 64"
python eval_rev.py --mode marigold $LOT_EV --tag lotusg_zs
python eval_rev.py --mode marigold $LOT_EV --ckpt runs/convin_lotusg_sd/final_trainable.pt --tag convin_lotusg_sd
python eval_rev.py --mode igle     $LOT_EV --ckpt runs/igle_lotusg_sd/final_trainable.pt   --tag igle_lotusg_sd

# 5. Per-image paired tests (Wilcoxon + bootstrap CI).
python analysis_rev.py stats --a marigold_sd      --b marigold_zs
python analysis_rev.py stats --a igle_sd          --b marigold_zs
python analysis_rev.py stats --a igle_sd          --b marigold_sd
python analysis_rev.py stats --a igle_sd          --b igle_v1
python analysis_rev.py stats --a convin_e2eft_sd  --b e2eft_zs
python analysis_rev.py stats --a igle_e2eft_sd    --b e2eft_zs
python analysis_rev.py stats --a igle_e2eft_sd    --b convin_e2eft_sd
python analysis_rev.py stats --a convin_lotusg_sd --b lotusg_zs
python analysis_rev.py stats --a igle_lotusg_sd   --b lotusg_zs
python analysis_rev.py stats --a igle_lotusg_sd   --b convin_lotusg_sd

# 6. External models (separate environment, see README). Alignment space is set per backend;
#    DA3 and DA3-Large are aligned in depth space as in the paper.
for m in dav2 dav2s daac daw e2eft robustdepth weatherdepth; do
  python eval_ext.py --model $m --tag ext_$m
done
for m in lotus_d lotus_g depthmaster; do python eval_ext.py --model $m --space disparity --tag ext_$m; done
for m in da3mono da3large moge2 depthpro; do python eval_ext.py --model $m --space depth --tag ext_$m; done

# 7. Benchmark, qualitative and mechanism figures, overview plot, bench.tex.
python figs_run.py
python figs_plot.py
