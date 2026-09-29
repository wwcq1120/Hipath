"""Inference: whole-image forward with optional ToMe token merging."""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from timm.layers import resample_abs_pos_embed
from torch.utils.data import DataLoader, Dataset

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SELF)

from config import RepairConfig
import repair_model as M

Image.MAX_IMAGE_PIXELS = None


def load_student(ckpt_path, device, eval_grid=0, conch_weight=""):
    cfg = RepairConfig()
    weight = conch_weight or cfg.conch_weight
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck_grid = ck["grid"]
    lora_r = ck.get("lora_r", cfg.lora_r)
    grid = eval_grid if eval_grid else ck_grid
    side = cfg.tile * grid
    sd = dict(ck["model"])
    if grid != ck_grid:
        g_old, g_new = ck_grid * cfg.gp, grid * cfg.gp
        sd["pos_embed"] = resample_abs_pos_embed(
            sd["pos_embed"], new_size=(g_new, g_new), old_size=(g_old, g_old),
            num_prefix_tokens=1, interpolation="bicubic", antialias=True,
        )
    student = M.build_student(
        device, side, gp=cfg.gp, lora_r=lora_r, conch_weight=weight
    )
    student.load_state_dict(sd, strict=True)
    student.eval()
    return student, grid, side, ck


def preprocess(path, side):
    im = Image.open(path).convert("RGB").resize((side, side), Image.BICUBIC)
    return TF.normalize(TF.to_tensor(im), M.CONCH_MEAN, M.CONCH_STD)


class ImgDS(Dataset):
    def __init__(self, items, side):
        self.items = items
        self.side = side

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        bag_id, path = self.items[i]
        try:
            return bag_id, preprocess(path, self.side)
        except Exception as e:
            print(f"[skip] {path} {e}", flush=True)
            return None


def collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch:
        return None
    ids = [x[0] for x in batch]
    xs = torch.stack([x[1] for x in batch])
    return ids, xs


@torch.no_grad()
def encode_batch(tome, xs, device, keep, feat):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = tome.forward(xs.to(device), keep)
    return z[:, 0:1] if feat == "cls" else z[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--split-csv", default=None, help="CSV with columns id,path")
    ap.add_argument("--out", default=None, help="output .pt for a single image")
    ap.add_argument("--out-dir", default=None, help="output directory for feats")
    ap.add_argument("--keep", type=float, default=0.2)
    ap.add_argument("--eval-grid", type=int, default=0)
    ap.add_argument("--feat", default="patch", choices=["patch", "cls"])
    ap.add_argument("--conch-weight", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    modes = [args.image, args.image_dir, args.split_csv]
    if sum(x is not None for x in modes) != 1:
        raise SystemExit("Specify exactly one of --image, --image-dir, --split-csv")

    device = f"cuda:{args.gpu}"
    student, grid, side, ck = load_student(
        args.ckpt, device, args.eval_grid, args.conch_weight or ""
    )
    tome = M.ToMeInfer(student)
    print(
        f"[load] grid={grid} side={side} keep={args.keep} "
        f"val_cos={ck.get('val_cos')}",
        flush=True,
    )

    if args.image:
        out = args.out or (os.path.splitext(args.image)[0] + ".pt")
        x = preprocess(args.image, side).unsqueeze(0)
        feat = encode_batch(tome, x, device, args.keep, args.feat)[0].half().cpu()
        torch.save(feat, out)
        print(f"[done] shape={tuple(feat.shape)} -> {out}", flush=True)
        return

    out_dir = args.out_dir or "./feats"
    feat_dir = os.path.join(out_dir, "feats") if args.split_csv else out_dir
    os.makedirs(feat_dir, exist_ok=True)

    if args.image_dir:
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        items = [
            (os.path.splitext(f)[0], os.path.join(args.image_dir, f))
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f.lower())[1] in exts
        ]
    else:
        df = pd.read_csv(args.split_csv)
        if "id" not in df.columns or "path" not in df.columns:
            raise ValueError("split.csv must contain columns: id, path")
        items = list(zip(df["id"].astype(str), df["path"].astype(str)))

    dl = DataLoader(
        ImgDS(items, side), batch_size=args.bs, shuffle=False,
        num_workers=4, collate_fn=collate,
    )
    done = 0
    for batch in dl:
        if batch is None:
            continue
        ids, xs = batch
        out_z = encode_batch(tome, xs, device, args.keep, args.feat)
        for b, bag_id in enumerate(ids):
            torch.save(out_z[b].half().cpu(), os.path.join(feat_dir, f"{bag_id}.pt"))
        done += len(ids)
        if done == len(ids) or done % 200 < args.bs:
            print(f"  {done}/{len(items)}", flush=True)
    print(f"[done] {done} features -> {feat_dir}", flush=True)


if __name__ == "__main__":
    main()
