"""Shared pieces for all scripts: model wrapper, inference and data helpers.

The base (Marigold v1-1, E2E-FT or Lotus-G) stays frozen; only the input path is trained.
Checkpoints store just the trainable modules, so every model is rebuilt from the frozen
base weights plus a small state dict.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextModel, CLIPTokenizer  # noqa: E402

from marigold.ramit_model.ramit import RAMiTCond, CondConvResidual  # noqa: E402
from marigold.ramit_model.depth_filter import SpatiallyVaryingDepthFilter  # noqa: E402

ROOT = os.environ.get("IGLE_ROOT", "/workspace/igledepth_v2")  # data, checkpoints and outputs
BASE_CKPT = f"{ROOT}/assets/checkpoints/marigold-depth-v1-1"
# Other SD2-UNet depth bases the module is plugged into (resolved from the HF cache).
BASES = {"marigold": BASE_CKPT, "e2eft": "GonzaloMG/marigold-e2e-ft-depth",
         "lotus_g": "jingheya/lotus-depth-g-v2-1-disparity"}
LATENT_SCALE = 0.18215
KB_H, KB_W = 352, 1216

# IGLEM configuration stated in the paper: RAMiT depths [2,1,1,2], dim 24, N=3 experts.
IGLEM_CFG = dict(dim=24, depths=(2, 1, 1, 2), num_heads=(4, 4, 4, 4), experts=3)


# --------------------------------------------------------------------------- model
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = (alpha or rank) / rank
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(x)) * self.scale


def add_lora(unet, rank):
    targets = ("to_q", "to_k", "to_v")
    n = 0
    for name, module in list(unet.named_modules()):
        for t in targets:
            child = getattr(module, t, None)
            if isinstance(child, nn.Linear):
                setattr(module, t, LoRALinear(child, rank))
                n += 1
        out = getattr(module, "to_out", None)
        if isinstance(out, nn.ModuleList) and isinstance(out[0], nn.Linear):
            out[0] = LoRALinear(out[0], rank)
            n += 1
    return n


def build_iglem():
    c = IGLEM_CFG
    adapter = RAMiTCond(dim=c["dim"], depths=c["depths"], num_heads=c["num_heads"])
    adapter.residual = CondConvResidual(lnt_dim=4, dim=c["dim"], K=c["experts"])
    nn.init.zeros_(adapter.residual.point_conv.weight)  # residual starts at zero
    return adapter


class DepthModel(nn.Module):
    """mode: 'marigold' (plain concat), 'igle' (IGLEM + SVDF), 'lora' (concat + UNet LoRA)."""

    def __init__(self, mode, lora_rank=0, base=BASE_CKPT, single_step=False):
        """single_step: E2E-FT-style base (deterministic, one step at t=T from a zero latent,
        trained on the clean-latent estimate) instead of Marigold's stochastic DDIM sampling."""
        super().__init__()
        self.mode = mode
        self.single_step = single_step
        base = BASES.get(base, base)
        self.vae = AutoencoderKL.from_pretrained(base, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet")
        tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder")
        # Marigold/E2E-FT encode the empty prompt unpadded; Lotus uses the standard SD
        # pipeline encoding padded to 77 tokens.
        pad = "max_length" if self.unet.config.class_embed_type == "projection" else "do_not_pad"
        with torch.no_grad():
            ids = tok("", padding=pad, max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids
            self.register_buffer("empty_text", te(ids)[0].detach(), persistent=False)
        del te
        sched_cfg = DDIMScheduler.load_config(base, subfolder="scheduler")
        self.train_sched = DDPMScheduler.from_config(
            sched_cfg, rescale_betas_zero_snr=not single_step, timestep_spacing="trailing")
        self.sched_cfg = sched_cfg
        # Lotus: x0 ("sample") prediction from a noise latent, plus a task embedding
        # (depth = [1, 0] -> sin/cos) fed through the UNet's projection class embedding.
        self.x0_pred = sched_cfg.get("prediction_type") == "sample"
        if self.unet.config.class_embed_type == "projection":
            te_ = torch.tensor([[1.0, 0.0]])
            self.register_buffer("task_emb", torch.cat([te_.sin(), te_.cos()], 1), persistent=False)
        else:
            self.task_emb = None
        self.adapter = None
        assert self.unet.config.in_channels == 8
        if mode == "igle":
            self.adapter = build_iglem()
            w, b = self.unet.conv_in.weight.clone(), self.unet.conv_in.bias.clone()
            svdf = SpatiallyVaryingDepthFilter(c_depth=4, c_guide=4, ksize=3, out_dim=320)
            svdf.new_conv_in.weight = nn.Parameter(w)
            svdf.new_conv_in.bias = nn.Parameter(b)
            self.unet.conv_in = svdf
        elif mode == "lora":
            assert lora_rank > 0
            add_lora(self.unet, lora_rank)
        elif mode != "marigold":
            raise ValueError(mode)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.trainable_modules().parameters():
            p.requires_grad_(True)

    def trainable_modules(self):
        mods = nn.ModuleDict({"conv_in": self.unet.conv_in})
        if self.adapter is not None:
            mods["adapter"] = self.adapter
        if self.mode == "lora":
            loras = nn.ModuleList([m for m in self.unet.modules() if isinstance(m, LoRALinear)])
            # base Linear stays frozen; only down/up are trained
            mods["lora"] = nn.ModuleList([nn.ModuleList([m.down, m.up]) for m in loras])
        return mods

    def trainable_state(self):
        return {k: v for k, v in self.state_dict().items()
                if k.startswith("unet.conv_in") or k.startswith("adapter.")
                or ".down.weight" in k or ".up.weight" in k}

    def load_trainable(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected
        return sd

    # ---------------- latents
    @torch.no_grad()
    def encode(self, img):  # img in [-1,1], [B,3,H,W]
        h = self.vae.encoder(img)
        mean, _ = torch.chunk(self.vae.quant_conv(h), 2, dim=1)
        return mean * LATENT_SCALE

    @torch.no_grad()
    def decode_depth(self, lat):
        z = self.vae.post_quant_conv(lat / LATENT_SCALE)
        return self.vae.decoder(z).mean(dim=1, keepdim=True)

    def guide(self, rgb, rgb_latent):
        if self.adapter is None:
            return rgb_latent
        return self.adapter(rgb, rgb_latent)

    def unet_forward(self, guide_latent, noisy, t):
        x = torch.cat([guide_latent, noisy], dim=1)
        emb = self.empty_text.to(x.dtype).expand(x.shape[0], -1, -1)
        cls = None if self.task_emb is None else self.task_emb.to(x.dtype).expand(x.shape[0], -1)
        return self.unet(x, t, encoder_hidden_states=emb, class_labels=cls).sample

    def single_step_x0(self, guide, generator=None):
        """Single-step bases at t=T. E2E-FT: zero latent, clean-latent estimate from v.
        Lotus-G: Gaussian noise latent, the UNet predicts the clean latent directly."""
        T = self.train_sched.config.num_train_timesteps - 1
        t = torch.full((guide.shape[0],), T, device=guide.device, dtype=torch.long)
        if self.x0_pred:
            lat = torch.randn(guide.shape, device=guide.device, dtype=guide.dtype, generator=generator)
            return self.unet_forward(guide, lat, t)
        lat = torch.zeros_like(guide)
        v = self.unet_forward(guide, lat, t)
        acp = self.train_sched.alphas_cumprod.to(guide.device)[T]
        return acp.sqrt() * lat - (1 - acp).sqrt() * v

    @torch.no_grad()
    def predict(self, rgb, steps=4, generator=None, return_latents=False):
        """rgb in [-1,1], H,W multiples of 8. Returns relative depth in [0,1], [B,1,H,W]."""
        rgb_lat = self.encode(rgb)
        guide = self.guide(rgb, rgb_lat)  # IGLEM runs once per image
        if self.single_step:
            lat = self.single_step_x0(guide.to(rgb_lat.dtype), generator)
        else:
            sched = DDIMScheduler.from_config(self.sched_cfg, rescale_betas_zero_snr=True,
                                              timestep_spacing="trailing")
            sched.set_timesteps(steps, device=rgb.device)
            lat = torch.randn(rgb_lat.shape, device=rgb.device, dtype=rgb_lat.dtype, generator=generator)
            for t in sched.timesteps:
                v = self.unet_forward(guide, lat, t)
                lat = sched.step(v, t, lat, generator=generator).prev_sample
        depth = ((self.decode_depth(lat).clamp(-1, 1) + 1) / 2)
        return (depth, guide) if return_latents else depth


# --------------------------------------------------------------------------- data utils
def read_rgb(path, size_wh=None):
    im = Image.open(path).convert("RGB")
    if size_wh is not None and im.size != tuple(size_wh):
        im = im.resize(tuple(size_wh), Image.BILINEAR)
    return np.asarray(im)


def kb_crop_box(h, w):
    top = h - KB_H
    left = (w - KB_W) // 2
    return top, left


def to_tensor_rgb(arr):
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0 * 2 - 1
    return t


def proc_size(h, w, proc_w):
    """Working resolution for bases that expect a fixed processing width (E2E-FT: 768)."""
    if not proc_w:
        return h, w
    return int(round(h * proc_w / w / 32)) * 32, proc_w


def resize_t(t, hw, mode="bilinear"):
    """Resize a [C,H,W] or [B,C,H,W] tensor; no-op if already that size."""
    if tuple(t.shape[-2:]) == tuple(hw):
        return t
    x = t[None] if t.dim() == 3 else t
    x = F.interpolate(x, size=hw, mode=mode, align_corners=False, antialias=mode == "bilinear")
    return x[0] if t.dim() == 3 else x


def pad_to_multiple(t, m=8):
    h, w = t.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="replicate")
    return t, h, w


def norm_depth(d, valid, q=0.01):
    """Marigold scale-shift normalisation to [-1,1] using 1%/99% quantiles of valid depth."""
    v = d[valid]
    lo, hi = torch.quantile(v, q), torch.quantile(v, 1 - q)
    out = (d - lo) / (hi - lo + 1e-6) * 2 - 1
    return out.clamp(-1, 1)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def count_params(module, trainable_only=False):
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

