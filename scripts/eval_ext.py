"""Zero-shot evaluation of recent external depth models under the revision protocol.

Every model sees exactly the same input region as our models (KB crop on KITTI, rows
232:776 on CADC, full frame on DrivingStereo) at its own recommended inference settings.
Outputs are aligned per image by least squares, in depth space for depth-like outputs and
in disparity space for disparity outputs, then scored with the same metrics, 80 m cap and
Eigen crop as eval_rev.py. Writes eval/<tag>/per_image_<set>.csv and summary.json.

  python eval_ext.py --model e2eft --tag ext_e2eft
  python eval_ext.py --model dav2  --tag ext_dav2
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
from eval_rev import MAN, MAX_D, prepare, to_gt_grid  # noqa: E402

ROOT = os.environ.get("IGLE_ROOT", "/workspace/igledepth_v2")  # data, checkpoints and outputs


def score(d, gt, mask):
    p, g = np.clip(d[mask], 1e-3, MAX_D), gt[mask]
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


def align(pred, gt, mask, space):
    p = pred[mask]
    A = np.stack([p, np.ones_like(p)], 1)
    if space == "depth":
        s, t = np.linalg.lstsq(A, gt[mask], rcond=None)[0]
        return s * pred + t
    s, t = np.linalg.lstsq(A, 1.0 / gt[mask], rcond=None)[0]
    return 1.0 / np.clip(s * pred + t, 1.0 / MAX_D, 1e3)


# ----------------------------------------------------------------------------- backends
# Each backend: load(dev) -> fn(PIL image) -> HxW numpy at the input size; plus output space.
def backend_e2eft(dev):
    from diffusers import MarigoldDepthPipeline
    pipe = MarigoldDepthPipeline.from_pretrained("GonzaloMG/marigold-e2e-ft-depth").to(dev)
    pipe.set_progress_bar_config(disable=True)
    # E2E-FT is deterministic: single trailing step at t = T with the noise latent set to zero
    orig = pipe.prepare_latents

    def zero_noise(*a, **k):
        image_latent, pred_latent = orig(*a, **k)
        return image_latent, torch.zeros_like(pred_latent)
    pipe.prepare_latents = zero_noise

    def fn(img):
        out = pipe(img, num_inference_steps=1, ensemble_size=1, processing_resolution=768,
                   generator=torch.Generator(device=dev).manual_seed(0))
        return np.asarray(out.prediction).squeeze().astype(np.float32)
    return fn, "depth"


def backend_dav2(dev, path=f"{ROOT}/assets/checkpoints/Depth-Anything-V2-Large-hf"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(path)
    model = AutoModelForDepthEstimation.from_pretrained(path).to(dev).eval()

    def fn(img):
        x = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            d = model(**x).predicted_depth
        return F.interpolate(d[:, None].float(), size=img.size[::-1], mode="bilinear",
                             align_corners=False)[0, 0].cpu().numpy()
    return fn, "disparity"


EXT = f"{ROOT}/repos/ext"


def _resize_to(pred, img):
    pred = np.asarray(pred, dtype=np.float32).squeeze()
    if pred.shape != img.size[::-1]:
        pred = F.interpolate(torch.from_numpy(pred)[None, None], size=img.size[::-1], mode="bilinear",
                             align_corners=False)[0, 0].numpy()
    return pred


def _fill_invalid(d):
    bad = ~np.isfinite(d)
    if bad.any():
        d = d.copy(); d[bad] = np.nanmax(np.where(bad, np.nan, d))
    return d


def backend_da3(dev, repo="depth-anything/DA3MONO-LARGE"):
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(repo).to(dev).eval()

    def fn(img):
        pred = model.inference([np.asarray(img)])
        return _resize_to(pred.depth[0], img)
    return fn, "depth"


def backend_moge2(dev):
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(dev).eval()

    def fn(img):
        x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1).to(dev)
        with torch.no_grad():
            out = model.infer(x)
        return _fill_invalid(out["depth"].float().cpu().numpy())
    return fn, "depth"


def backend_depthpro(dev):
    from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
    proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch.float16).to(dev).eval()

    def fn(img):
        x = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(pixel_values=x["pixel_values"].half())
        d = proc.post_process_depth_estimation(out, target_sizes=[img.size[::-1]])[0]["predicted_depth"]
        return d.float().cpu().numpy()
    return fn, "depth"


def backend_lotus(dev, repo="jingheya/lotus-depth-d-v2-0-disparity", mode="regression"):
    import importlib.util
    sys.path.insert(0, f"{EXT}/Lotus")
    spec = importlib.util.spec_from_file_location("lotus_pipeline", f"{EXT}/Lotus/pipeline.py")
    lp = importlib.util.module_from_spec(spec); spec.loader.exec_module(lp)
    cls = lp.LotusDPipeline if mode == "regression" else lp.LotusGPipeline
    pipe = cls.from_pretrained(repo, torch_dtype=torch.float16).to(dev)
    pipe.set_progress_bar_config(disable=True)
    task_emb = torch.tensor([1, 0]).float().unsqueeze(0).to(dev)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    def fn(img):
        x = torch.from_numpy(np.asarray(img).astype(np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad(), torch.autocast("cuda"):
            pred = pipe(rgb_in=x, prompt="", num_inference_steps=1, generator=torch.Generator(device=dev).manual_seed(42),
                        output_type="np", timesteps=[999], task_emb=task_emb, processing_res=768,
                        match_input_res=True, resample_method="bilinear").images[0]
        return _resize_to(pred.mean(axis=-1), img)
    return fn, "disparity"


def backend_depthmaster(dev):
    from huggingface_hub import snapshot_download
    sys.path.insert(0, f"{EXT}/DepthMaster")
    from depthmaster import DepthMasterPipeline
    from depthmaster.modules.unet_2d_condition_s2 import UNet2DConditionModel
    ck = snapshot_download("zysong212/DepthMaster")
    pipe = DepthMasterPipeline.from_pretrained(ck, torch_dtype=torch.float32)
    pipe.unet = UNet2DConditionModel.from_pretrained(os.path.join(ck, "unet"))
    pipe = pipe.to(dev)

    def fn(img):
        with torch.no_grad():
            out = pipe(img, processing_res=768, match_input_res=True, resample_method="bilinear",
                       color_map=None, show_progress_bar=False)
        return _resize_to(out.depth_np, img)
    return fn, "depth"


def backend_monovit(dev, repo="WeatherDepth", pkg="networks", ckpt="weatherdepthVit"):
    """Weather-robust self-supervised MonoViT models (WeatherDepth*, Robust-Depth ViT):
    MPViT-small encoder + HR decoder at 640x192, [0,1] input, sigmoid disparity output."""
    import importlib
    import types
    # The MPViT file imports mmcv/mmseg only for ImageNet-init loading (overridden by
    # encoder.pth), backbone registration and BN construction; stub those few names.
    def mod(name, **attrs):
        m = sys.modules.setdefault(name, types.ModuleType(name))
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def build_norm_layer(cfg, n):
        t = cfg.get("type", "BN")
        return t.lower(), (torch.nn.LayerNorm(n) if t == "LN" else torch.nn.BatchNorm2d(n))

    class _Reg:
        def register_module(self, *a, **k):
            return a[0] if a and callable(a[0]) else (lambda c: c)
    noop = lambda *a, **k: None  # noqa: E731
    mod("mmcv", runner=mod("mmcv.runner", load_checkpoint=noop, load_state_dict=noop),
        cnn=mod("mmcv.cnn", build_norm_layer=build_norm_layer))
    mod("mmseg", utils=mod("mmseg.utils", get_root_logger=noop),
        models=mod("mmseg.models", builder=mod("mmseg.models.builder", BACKBONES=_Reg())))
    wdir = f"{ROOT}/assets/weather_ckpt/{ckpt}"
    run_dir = f"{ROOT}/tmp/monovit_{ckpt}"
    os.makedirs(f"{run_dir}/ckpt", exist_ok=True)
    link = f"{run_dir}/ckpt/mpvit_small.pth"
    if not os.path.exists(link):
        os.symlink(f"{ROOT}/assets/weather_ckpt/weatherdepthVit/mpvit_small.pth", link)
    sys.path.insert(0, f"{EXT}/{repo}")
    mpvit = importlib.import_module(f"{pkg}.mpvit")
    hrd = importlib.import_module(f"{pkg}.hr_decoder")
    cwd = os.getcwd(); os.chdir(run_dir)  # mpvit_small() reads ./ckpt/mpvit_small.pth
    enc = mpvit.mpvit_small()
    os.chdir(cwd)
    enc.num_ch_enc = [64, 128, 216, 288, 288]
    dec = hrd.HR_DepthDecoder() if hasattr(hrd, "HR_DepthDecoder") else hrd.DepthDecoder()
    ed = torch.load(f"{wdir}/encoder.pth", map_location="cpu")
    md = enc.state_dict()
    miss = [k for k in md if k not in ed]
    enc.load_state_dict({k: v for k, v in ed.items() if k in md}, strict=False)
    dec.load_state_dict(torch.load(f"{wdir}/depth.pth", map_location="cpu"))
    print(f"{ckpt}: encoder keys missing from ckpt: {len(miss)}", flush=True)
    enc.to(dev).eval(); dec.to(dev).eval()  # MPViT.train() returns None, so no chaining

    def fn(img):
        x = np.asarray(img.resize((640, 192), Image.LANCZOS), dtype=np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
        with torch.no_grad():
            disp = dec(enc(x))[("disp", 0)]
        return _resize_to(disp[0, 0].float().cpu().numpy(), img)
    return fn, "disparity"


def backend_daac(dev):
    """Depth Anything AC (ViT-S, fine-tuned from DAv2 for adverse conditions): shorter side
    518, sides rounded to multiples of 14, ImageNet normalisation, disparity output."""
    sys.path.insert(0, f"{EXT}/DepthAnythingAC")
    from depth_anything.dpt import DepthAnything_AC
    # The DINOv2 hub entry reads ./checkpoints/dinov2_vits14/... and ./torchhub relative to
    # the cwd; build from a run dir that links both (weights are then overwritten anyway).
    ck = f"{ROOT}/assets/weather_ckpt/daac/checkpoints"
    run_dir = f"{ROOT}/tmp/daac_run"
    os.makedirs(run_dir, exist_ok=True)
    for name, src in [("checkpoints", ck), ("torchhub", f"{EXT}/DepthAnythingAC/torchhub")]:
        if not os.path.exists(f"{run_dir}/{name}"):
            os.symlink(src, f"{run_dir}/{name}")
    cwd = os.getcwd(); os.chdir(run_dir)
    model = DepthAnything_AC({"encoder": "vits", "features": 64,
                              "out_channels": [48, 96, 192, 384], "version": "v2"})
    os.chdir(cwd)
    path = f"{ck}/depth_anything_AC_vits.pth"
    miss, unexp = model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
    print(f"daac {path}: missing {len(miss)} unexpected {len(unexp)}", flush=True)
    model.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def fn(img):
        w, h = img.size
        s = 518 / min(h, w)
        nh, nw = int(round(h * s / 14)) * 14, int(round(w * s / 14)) * 14
        x = np.asarray(img.resize((nw, nh), Image.BICUBIC), dtype=np.float32) / 255.0
        x = (torch.from_numpy(x).permute(2, 0, 1)[None].to(dev) - mean) / std
        with torch.no_grad():
            d = model(x)["out"]
        return _resize_to(d.float().squeeze().cpu().numpy(), img)
    return fn, "disparity"


def backend_daw(dev):
    """DA-W (ViT-S DAv2 with AdaLN weather conditioning from a Style Filter), using the
    repo's own 518 lower-bound / multiple-of-14 transform; disparity output."""
    sys.path.insert(0, f"{EXT}/DA-W")
    from infer import build_transform, load_daw, run_forward
    ck = f"{ROOT}/assets/weather_ckpt/daw"
    model, sf = load_daw(f"{ck}/daw_vits_stage2.pth", f"{ck}/daw_style_filter_stage1.pth", dev)
    tf = build_transform(518)

    def fn(img):
        x = tf({"image": np.asarray(img, dtype=np.float32) / 255.0})["image"]
        d = run_forward(model, torch.from_numpy(x)[None].to(dev), sf)
        return _resize_to(d.float().squeeze().cpu().numpy(), img)
    return fn, "disparity"


BACKENDS = {"weatherdepth": backend_monovit, "daac": backend_daac, "daw": backend_daw,
            "dav2s": lambda d: backend_dav2(d, "depth-anything/Depth-Anything-V2-Small-hf"),
            "robustdepth": lambda d: backend_monovit(d, "Robust-Depth/Robust-Depth", "networksvit",
                                                     "robustdepthVit"),
            "e2eft": backend_e2eft, "dav2": backend_dav2, "da3mono": backend_da3,
            "da3large": lambda d: backend_da3(d, "depth-anything/DA3-LARGE"),
            "moge2": backend_moge2, "depthpro": backend_depthpro, "lotus_d": backend_lotus,
            "lotus_g": lambda d: backend_lotus(d, "jingheya/lotus-depth-g-v2-1-disparity", "generation"),
            "depthmaster": backend_depthmaster}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sets", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--space", default=None, help="override alignment space (depth|disparity)")
    args = ap.parse_args()
    dev = "cuda"
    fn, space = BACKENDS[args.model](dev)
    space = args.space or space

    sets = [os.path.basename(f)[5:-5] for f in sorted(glob.glob(f"{MAN}/test_*.json"))]
    if args.sets != "all":
        sets = [s for s in sets if s in args.sets.split(",")]
    out_dir = f"{ROOT}/revision/eval/{args.tag}"
    os.makedirs(out_dir, exist_ok=True)
    summary = {"args": vars(args), "align_space": space}
    for set_name in sets:
        samples = json.load(open(f"{MAN}/test_{set_name}.json"))["samples"]
        if args.limit:
            samples = samples[: args.limit]
        rows, times = [], []
        for s in tqdm(samples, desc=f"{args.tag}:{set_name}"):
            inp, gt, mask, region = prepare(s, set_name)
            img = Image.fromarray(((inp.permute(1, 2, 0).numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8))
            torch.cuda.synchronize(); t0 = time.time()
            pred = fn(img)
            torch.cuda.synchronize(); times.append(time.time() - t0)
            if pred.shape != img.size[::-1]:
                pred = F.interpolate(torch.from_numpy(pred)[None, None], size=img.size[::-1],
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
            full = to_gt_grid(pred.astype(np.float32), region, gt.shape)
            rows.append(dict(id=s["sample_id"], **score(align(full, gt, mask, space), gt, mask)))
        with open(f"{out_dir}/per_image_{set_name}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
        mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"}
        mean["sec_per_image"] = float(np.median(times[1:] if len(times) > 1 else times))
        mean["n"] = len(rows)
        summary[set_name] = mean
        print(set_name, {k: round(v, 4) for k, v in mean.items()}, flush=True)
    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()

