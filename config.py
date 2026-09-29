"""Training and model configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


_PKG = os.path.dirname(os.path.abspath(__file__))


@dataclass
class RepairConfig:
    # Path to CONCH OpenCLIP checkpoint (pytorch_model.bin).
    # Set via environment variable CONCH_WEIGHT or --conch-weight.
    conch_weight: str = field(
        default_factory=lambda: os.environ.get("CONCH_WEIGHT", "")
    )
    # Root directory of unlabeled training images (class subfolders).
    # Set via environment variable TRAIN_DATA_ROOT or --data-root.
    tcga_root: str = field(
        default_factory=lambda: os.environ.get("TRAIN_DATA_ROOT", "")
    )

    grid: int = 2
    tile: int = 448
    patch: int = 16
    lora_r: int = 8

    bs: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_steps: int = 4000
    warmup_frac: float = 0.03
    n_val: int = 200
    eval_every: int = 100
    patience: int = 6
    min_delta: float = 1e-3
    num_workers: int = 12
    seed: int = 2024

    out_dir: str = field(default_factory=lambda: os.path.join(_PKG, "weights", "g2"))

    @property
    def side(self) -> int:
        return self.tile * self.grid

    @property
    def gp(self) -> int:
        return self.tile // self.patch
