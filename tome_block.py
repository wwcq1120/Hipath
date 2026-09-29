"""ToMe block utilities used at inference time."""
from __future__ import annotations

import torch
from timm.models.vision_transformer import Block

from tome.merge import bipartite_soft_matching, merge_wavg


def build_r_schedule(
    n_tokens: int,
    depth: int,
    target_k: int,
    no_merge_last: int,
    no_merge_first: int = 0,
):
    """Per-layer token removal schedule approaching target_k."""
    r = [0] * depth
    merge_idx = list(range(no_merge_first, depth - no_merge_last))
    merge_blocks = max(1, len(merge_idx))
    cur = n_tokens
    for j, i in enumerate(merge_idx):
        remaining = merge_blocks - j
        target_after = max(
            target_k, int(round(cur * (target_k / cur) ** (1.0 / remaining)))
        )
        rem = min(cur - target_after, cur // 2)
        r[i] = max(rem, 0)
        cur -= r[i]
    return r


class TrackedToMeBlock(Block):
    """timm Block with bipartite ToMe between attention and MLP."""

    def _dp1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _dp2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        info = self._tome_info
        attn_size = info["size"] if info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        if hasattr(self, "ls1"):
            x_attn = self.ls1(x_attn)
        x = x + self._dp1(x_attn)

        r = info["r"].pop(0)
        if r > 0:
            merge, unmerge = bipartite_soft_matching(
                metric, r, info["class_token"], info["distill_token"]
            )
            info["unmerge_list"].append(unmerge)
            x, info["size"] = merge_wavg(merge, x, info["size"])

        mlp_out = self.mlp(self.norm2(x))
        if hasattr(self, "ls2"):
            mlp_out = self.ls2(mlp_out)
        x = x + self._dp2(mlp_out)

        info["block_counter"] += 1
        if info["block_counter"] == info.get("no_merge_first", 0):
            n_prefix = 1 if info["class_token"] else 0
            info["first_full"] = x[:, n_prefix:].detach()
        return x
