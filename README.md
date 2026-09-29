# Supplementary Code and Checkpoints

This package provides the implementation and pretrained adapters for high-resolution pathology foundation model adaptation with token merging, accompanying the main paper.

## Contents

```text
supplementary/
├── README.md
├── requirements.txt
├── config.py
├── repair_model.py
├── tome_block.py
├── train.py
├── infer.py
├── tome/                 # Token Merging (ToMe) operators
└── weights/
    ├── checkpoints.json
    ├── g2/best.pt        # input 896×896
    ├── g3/best.pt        # input 1344×1344
    ├── g4/best.pt        # input 1792×1792
    └── g5/best.pt        # input 2240×2240
```

## Method Overview

Starting from a frozen CONCH vision backbone (ViT-B/16, native tile size \(448\times448\)):

1. Resize each ROI to \(448G \times 448G\), where \(G\) is the adaptation grid.
2. Train a lightweight adapter on unlabeled images by distilling in-distribution tile features into a single whole-image forward pass. Trainable modules are absolute positional embeddings, LayerNorm parameters, and LoRA adapters on attention `qkv` / `proj` (rank \(r=8\)).
3. At inference, apply bipartite Token Merging (ToMe) to retain a fraction \(\rho\) of patch tokens (default \(\rho=0.2\)).

The optimization objective is patch-wise cosine alignment between the student whole-image features and the teacher tile-grid features.

## Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch with CUDA, and a GPU supporting `bfloat16` autocast.

Obtain the official CONCH checkpoint (`pytorch_model.bin`) and set:

```bash
export CONCH_WEIGHT=/path/to/pytorch_model.bin
```

## Checkpoints

| Checkpoint | Grid \(G\) | Input size | Patch tokens (before ToMe) | Validation cosine |
|------------|------------|------------|----------------------------|-------------------|
| `weights/g2/best.pt` | 2 | \(896\times896\) | 3136 | 0.9829 |
| `weights/g3/best.pt` | 3 | \(1344\times1344\) | 7056 | 0.9775 |
| `weights/g4/best.pt` | 4 | \(1792\times1792\) | 12544 | 0.9692 |
| `weights/g5/best.pt` | 5 | \(2240\times2240\) | 19600 | 0.9575 |

Each file stores `model` (state dict), `grid`, `side`, `lora_r`, `step`, and `val_cos`. See `weights/checkpoints.json` for a machine-readable summary.

**Usage guidance.** Prefer `g2` for ROIs near \(896^2\), and `g4` for larger ROIs near \(1792^2\). When the evaluation resolution differs from the training grid, pass `--eval-grid` to interpolate positional embeddings.

## Training

Training images should be organized as `DATA_ROOT/<subdir>/*.{jpg,png,tif,...}`.

```bash
export CONCH_WEIGHT=/path/to/pytorch_model.bin
export TRAIN_DATA_ROOT=/path/to/unlabeled_images

CUDA_VISIBLE_DEVICES=0 python train.py \
  --grid 4 \
  --data-root "$TRAIN_DATA_ROOT" \
  --conch-weight "$CONCH_WEIGHT" \
  --out weights/g4
```

Key arguments: `--grid`, `--bs`, `--lr`, `--max-steps`, `--patience`.

Images are resized to \(448G\), preloaded into memory, and optimized with AdamW and cosine learning-rate decay after a short warmup. Early stopping monitors held-out validation cosine similarity.

## Inference

### Single image

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --ckpt weights/g4/best.pt \
  --image /path/to/roi.jpg \
  --out feat.pt \
  --keep 0.2 \
  --conch-weight "$CONCH_WEIGHT"
```

Output `feat.pt` is a `float16` tensor of shape `(K, 768)` (patch tokens). Use `--feat cls` to save only the class token `(1, 768)`.

### Image directory

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --ckpt weights/g4/best.pt \
  --image-dir /path/to/images \
  --out-dir /path/to/feats \
  --keep 0.2 \
  --conch-weight "$CONCH_WEIGHT"
```

### CSV list (`id,path`)

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --ckpt weights/g4/best.pt \
  --split-csv /path/to/split.csv \
  --out-dir /path/to/output \
  --keep 0.2 \
  --conch-weight "$CONCH_WEIGHT"
```

Features are written to `output/feats/{id}.pt`.

### Common options

| Argument | Description |
|----------|-------------|
| `--keep` | Token keep ratio \(\rho\) after ToMe (default `0.2`; `1.0` disables merging) |
| `--eval-grid` | Evaluate at a grid different from the checkpoint training grid |
| `--feat` | `patch` (default) or `cls` |
| `--bs` | Batch size |

## Recommended Protocol

| Item | Setting |
|------|---------|
| Backbone | CONCH ViT-B/16 |
| Adapter | positional embeddings + LayerNorm + LoRA |
| Default keep ratio | \(\rho = 0.2\) |
| Feature type | patch tokens |
| Downstream head | Attention-based MIL (ABMIL), mean-pool linear classifier, or \(k\)NN as in the paper |

## Citation

Please cite the main paper when using this code or the provided checkpoints. Token Merging follows [Bolya et al., ToMe](https://github.com/facebookresearch/ToMe). CONCH is used under its original license and should be obtained from the official release.
