from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class TFCDeletionContrast(nn.Module):
    def __init__(self, margin: float = 0.02) -> None:
        super().__init__()
        self.margin = float(margin)
        self.register_buffer("ema_background", torch.zeros(1), persistent=False)

    def _replace(self, patch_tokens: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        b, layers, n, d = patch_tokens.shape
        patched = patch_tokens.clone()
        flat = patched.reshape(b, layers * n, d)
        bg = flat.mean(dim=1, keepdim=True)
        for i in range(b):
            idx = flat_idx[i].unique().clamp(0, layers * n - 1)
            flat[i, idx] = bg[i]
        return flat.reshape_as(patch_tokens)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        topk_indices: torch.Tensor,
        credit_norm: torch.Tensor,
        head_fn: Callable[[torch.Tensor], torch.Tensor],
        target_logits: torch.Tensor,
        target_labels: torch.Tensor | None = None,
        max_factors_per_sample: int = 4,
        same_region_background: str = "ema",
    ) -> dict:
        b, factors, k = topk_indices.shape
        targets = credit_norm.shape[-1]
        device = patch_tokens.device
        selected_effect = torch.zeros(b, targets, device=device, dtype=patch_tokens.dtype)
        random_effect = torch.zeros_like(selected_effect)
        selected_mask = torch.zeros_like(selected_effect, dtype=torch.bool)
        for i in range(b):
            score = credit_norm[i].abs()
            flat_order = torch.argsort(score.flatten(), descending=True)
            used = 0
            for flat_pos in flat_order.tolist():
                f = flat_pos // targets
                t = flat_pos % targets
                if used >= max_factors_per_sample:
                    break
                if score[f, t] <= 0:
                    continue
                sel_idx = topk_indices[i, f].view(1, -1)
                rand_pool = torch.randperm(topk_indices.shape[-1] * max(1, factors), device=device)[: sel_idx.numel()]
                rand_idx = rand_pool.remainder(patch_tokens.shape[1] * patch_tokens.shape[2]).view(1, -1)
                patched_sel = self._replace(patch_tokens[i : i + 1], sel_idx)
                patched_rand = self._replace(patch_tokens[i : i + 1], rand_idx)
                sel_logits = head_fn(patched_sel)
                rnd_logits = head_fn(patched_rand)
                selected_effect[i, t] = target_logits[i, t] - sel_logits[0, t]
                random_effect[i, t] = target_logits[i, t] - rnd_logits[0, t]
                selected_mask[i, t] = True
                used += 1
        gap = selected_effect - random_effect
        valid = selected_mask
        if bool(valid.any()):
            loss = F.relu(self.margin - gap[valid]).mean()
            rate = (gap[valid] > 0).float().mean()
            mean_gap = gap[valid].mean()
        else:
            loss = patch_tokens.new_tensor(0.0)
            rate = patch_tokens.new_tensor(0.0)
            mean_gap = patch_tokens.new_tensor(0.0)
        return {
            "selected_effect": selected_effect,
            "random_effect": random_effect,
            "deletion_contrast_loss": loss,
            "selected_gt_random_rate": rate,
            "selected_gt_random_mask": gap > 0,
            "selected_vs_random_gap": gap,
            "target_flip_rate": (gap.abs() > 0.5).float().mean(),
            "stats": {
                "selected_vs_random_gap_mean": float(mean_gap.detach().cpu()),
                "selected_gt_random_rate": float(rate.detach().cpu()),
                "valid_pairs": int(valid.sum().detach().cpu()),
            },
        }
