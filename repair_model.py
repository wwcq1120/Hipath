"""High-resolution adaptation model: frozen CONCH + trainable pos/LN/LoRA + ToMe inference."""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Attention, Block

_PKG = os.path.dirname(os.path.abspath(__file__))
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from tome_block import TrackedToMeBlock, build_r_schedule  # noqa: E402
from tome.patch.timm import ToMeAttention  # noqa: E402

CONCH_MEAN = (0.48145466, 0.4578275, 0.40821073)
CONCH_STD = (0.26862954, 0.26130258, 0.27577711)


def conch_visual_sd(weight: str):
    if not weight or not os.path.isfile(weight):
        raise FileNotFoundError(
            "CONCH weight not found. Set CONCH_WEIGHT or pass --conch-weight "
            "pointing to pytorch_model.bin"
        )
    sd = torch.load(weight, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    p = "visual.trunk."
    return {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t()) @ self.B.t() * self.scale


def inject_lora(model, r=8):
    for blk in model.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, r)
        blk.attn.proj = LoRALinear(blk.attn.proj, r)


def build_teacher(device, tile=448, conch_weight: str = ""):
    m = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0,
        img_size=tile, dynamic_img_size=False,
    )
    m.load_state_dict(conch_visual_sd(conch_weight), strict=False)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def build_student(device, side, gp=28, lora_r=8, conch_weight: str = ""):
    m = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0,
        img_size=side, dynamic_img_size=False,
    )
    sd = conch_visual_sd(conch_weight)
    g_new = side // 16
    sd["pos_embed"] = resample_abs_pos_embed(
        sd["pos_embed"], new_size=(g_new, g_new), old_size=(gp, gp),
        num_prefix_tokens=1, interpolation="bicubic", antialias=True,
    )
    m.load_state_dict(sd, strict=False)
    for n, p in m.named_parameters():
        p.requires_grad_(("pos_embed" in n) or ("norm" in n))
    inject_lora(m, lora_r)
    m.to(device)
    return m


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def count_trainable(model):
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntot = sum(p.numel() for p in model.parameters())
    return ntr, ntot


@torch.no_grad()
def teacher_grid(teacher, imgs, grid, device, tile=448, gp=28, chunk=64):
    B = imgs.shape[0]
    tiles = []
    for i in range(grid):
        for j in range(grid):
            tiles.append(
                imgs[:, :, i * tile:(i + 1) * tile, j * tile:(j + 1) * tile]
            )
    tiles = torch.stack(tiles, 1).reshape(B * grid * grid, 3, tile, tile).to(device)
    out = []
    for s in range(0, tiles.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = teacher.forward_features(tiles[s:s + chunk])[:, 1:]
        out.append(f.float())
    patch = torch.cat(out, 0).reshape(B, grid, grid, gp, gp, -1)
    C = patch.shape[-1]
    big = patch.permute(0, 1, 3, 2, 4, 5).reshape(B, grid * gp, grid * gp, C)
    return big.reshape(B, (grid * gp) ** 2, C)


def student_grid(student, imgs, device):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = student.forward_features(imgs.to(device))[:, 1:]
    return f.float()


def align_cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).mean()


class ToMeInfer:
    def __init__(self, student):
        self.m = student
        for mod in student.modules():
            if isinstance(mod, Block):
                mod.__class__ = TrackedToMeBlock
            elif isinstance(mod, Attention):
                mod.__class__ = ToMeAttention
        self.info = {
            "r": [], "size": None, "unmerge_list": [], "prop_attn": False,
            "class_token": True, "distill_token": False,
            "no_merge_first": 0, "block_counter": 0, "first_full": None,
        }
        for mod in student.modules():
            if isinstance(mod, TrackedToMeBlock):
                mod._tome_info = self.info

    @torch.no_grad()
    def forward(self, x, keep):
        m = self.m
        B = x.shape[0]
        tok = m.patch_embed(x)
        n_patch = tok.shape[1]
        cls = m.cls_token.expand(B, -1, -1)
        z = torch.cat([cls, tok], 1) + m.pos_embed
        z = m.pos_drop(z)
        target_k = max(1, int(round(keep * n_patch)))
        self.info["r"] = build_r_schedule(n_patch, len(m.blocks), target_k, 0, 0)
        self.info["size"] = None
        self.info["unmerge_list"] = []
        self.info["block_counter"] = 0
        self.info["first_full"] = None
        for blk in m.blocks:
            z = blk(z)
        z = m.norm(z)
        return z.float()
