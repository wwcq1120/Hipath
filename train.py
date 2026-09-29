"""Train the high-resolution adapter with unlabeled tile-to-image distillation."""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_SELF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SELF)

from config import RepairConfig
import repair_model as M

Image.MAX_IMAGE_PIXELS = None


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def list_images(root):
    paths = []
    for c in sorted(os.listdir(root)):
        cd = os.path.join(root, c)
        if not os.path.isdir(cd):
            continue
        for f in os.listdir(cd):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                paths.append(os.path.join(cd, f))
    return paths


class PreloadDS(Dataset):
    def __init__(self, paths, side):
        self.paths = paths
        self.side = side

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB").resize(
                (self.side, self.side), Image.BICUBIC
            )
        except Exception:
            return (i, None)
        return (i, TF.pil_to_tensor(im))


def preload(paths, side, workers):
    pool = torch.zeros(len(paths), 3, side, side, dtype=torch.uint8)
    ok = torch.zeros(len(paths), dtype=torch.bool)
    dl = DataLoader(
        PreloadDS(paths, side), batch_size=1, shuffle=False,
        num_workers=workers, collate_fn=lambda b: b[0],
    )
    t0 = time.time()
    for k, (i, t) in enumerate(dl):
        if t is not None:
            pool[i] = t
            ok[i] = True
        if (k + 1) % 500 == 0:
            print(f"  preload {k + 1}/{len(paths)} ({time.time() - t0:.0f}s)", flush=True)
    idx = ok.nonzero(as_tuple=True)[0].tolist()
    print(f"[preload] {len(idx)}/{len(paths)} @ {side}px, {time.time() - t0:.0f}s", flush=True)
    return pool, idx


def norm_batch(pool, idx, device):
    x = pool[idx].float().div_(255.).to(device)
    return TF.normalize(x, M.CONCH_MEAN, M.CONCH_STD)


@torch.no_grad()
def eval_val(student, teacher, pool, val_idx, grid, device, gp, bs=8):
    student.eval()
    cs = []
    for s in range(0, len(val_idx), bs):
        idx = val_idx[s:s + bs]
        imgs = norm_batch(pool, idx, device)
        tg = M.teacher_grid(teacher, imgs, grid, device, gp=gp)
        sg = M.student_grid(student, imgs, device)
        cs.append(M.align_cos(sg, tg).item())
    return float(np.mean(cs))


def main():
    cfg = RepairConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--grid", type=int, default=cfg.grid)
    ap.add_argument("--bs", type=int, default=cfg.bs)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--max-steps", type=int, default=cfg.max_steps)
    ap.add_argument("--patience", type=int, default=cfg.patience)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--conch-weight", default=None)
    ap.add_argument("--max", type=int, default=0)
    args = ap.parse_args()

    if args.data_root:
        cfg.tcga_root = args.data_root
    if args.conch_weight:
        cfg.conch_weight = args.conch_weight

    set_seed(cfg.seed)
    device = f"cuda:{args.gpu}"
    grid = args.grid
    side = cfg.tile * grid
    out = args.out or os.path.join(_SELF, "weights", f"g{grid}")
    os.makedirs(out, exist_ok=True)

    if not cfg.conch_weight or not os.path.isfile(cfg.conch_weight):
        raise FileNotFoundError("Please provide CONCH weights via --conch-weight or CONCH_WEIGHT")
    if not cfg.tcga_root or not os.path.isdir(cfg.tcga_root):
        raise FileNotFoundError("Please provide training image root via --data-root or TRAIN_DATA_ROOT")

    paths = list_images(cfg.tcga_root)
    random.Random(cfg.seed).shuffle(paths)
    if args.max:
        paths = paths[: args.max]
    print(f"[data] {len(paths)} images, side={side}, grid={grid}", flush=True)

    pool, idx = preload(paths, side, cfg.num_workers)
    random.Random(cfg.seed).shuffle(idx)
    n_val = min(cfg.n_val, max(1, len(idx) // 5))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"[split] train={len(tr_idx)} val={len(val_idx)}", flush=True)

    teacher = M.build_teacher(device, conch_weight=cfg.conch_weight)
    student = M.build_student(
        device, side, gp=cfg.gp, lora_r=cfg.lora_r, conch_weight=cfg.conch_weight
    )
    student.set_grad_checkpointing(True)
    ntr, ntot = M.count_trainable(student)
    print(
        f"[student] trainable {ntr / 1e6:.3f}M / {ntot / 1e6:.1f}M ({100 * ntr / ntot:.2f}%)",
        flush=True,
    )

    opt = torch.optim.AdamW(
        M.trainable_params(student), lr=args.lr, weight_decay=cfg.weight_decay
    )
    warmup = int(cfg.warmup_frac * args.max_steps)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, args.max_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * p))

    best = (-1.0, -1)
    no_improve = 0
    t0 = time.time()
    rng = random.Random(cfg.seed)
    step = 0
    while step < args.max_steps:
        step += 1
        for g in opt.param_groups:
            g["lr"] = args.lr * lr_at(step)
        bidx = [rng.choice(tr_idx) for _ in range(args.bs)]
        imgs = norm_batch(pool, bidx, device)
        student.train()
        tg = M.teacher_grid(teacher, imgs, grid, device, gp=cfg.gp)
        sg = M.student_grid(student, imgs, device)
        loss = 1.0 - M.align_cos(sg, tg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(
                f"  step {step}/{args.max_steps} train_cos={1 - loss.item():.4f} "
                f"lr={opt.param_groups[0]['lr']:.2e}",
                flush=True,
            )
        if step % cfg.eval_every == 0:
            vc = eval_val(
                student, teacher, pool, val_idx, grid, device, cfg.gp, bs=args.bs
            )
            improved = vc > best[0] + cfg.min_delta
            print(
                f"  >> step {step} val_cos={vc:.4f} best={max(vc, best[0]):.4f}"
                f"{' *' if improved else ''}",
                flush=True,
            )
            if improved:
                best = (vc, step)
                no_improve = 0
                torch.save(
                    {
                        "model": student.state_dict(),
                        "grid": grid,
                        "side": side,
                        "lora_r": cfg.lora_r,
                        "step": step,
                        "val_cos": vc,
                    },
                    os.path.join(out, "best.pt"),
                )
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(
                        f"[early-stop] best val_cos={best[0]:.4f}@{best[1]}",
                        flush=True,
                    )
                    break

    torch.save(
        {
            "model": student.state_dict(),
            "grid": grid,
            "side": side,
            "lora_r": cfg.lora_r,
            "step": step,
        },
        os.path.join(out, "last.pt"),
    )
    print(
        f"[done] best val_cos={best[0]:.4f}@step{best[1]}, "
        f"{(time.time() - t0) / 60:.1f} min -> {out}/best.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
