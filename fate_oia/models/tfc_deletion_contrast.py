from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


class TFCDeletionContrast(nn.Module):
    def __init__(self, margin: float = 0.02, ema_momentum: float = 0.95) -> None:
        super().__init__()
        self.margin = float(margin)
        self.ema_momentum = float(ema_momentum)
        self.register_buffer("ema_background", torch.empty(0), persistent=False)

    def _background(
        self,
        flat: torch.Tensor,
        sample_idx: int,
        bg_pool: torch.Tensor,
        same_region_background: str,
    ) -> torch.Tensor:
        mode = str(same_region_background).lower()
        if bg_pool.numel() == 0:
            current = flat[sample_idx].mean(dim=0, keepdim=True)
        else:
            current = flat[sample_idx, bg_pool].mean(dim=0, keepdim=True)

        if mode == "ema":
            current_detached = current.detach()
            if not self.training:
                if self.ema_background.numel() == current_detached.numel() and self.ema_background.device == current_detached.device:
                    return self.ema_background.to(dtype=flat.dtype)
                return current_detached
            if self.ema_background.numel() != current_detached.numel() or self.ema_background.device != current_detached.device:
                self.ema_background = current_detached.clone()
            else:
                self.ema_background.mul_(self.ema_momentum).add_(current_detached, alpha=1.0 - self.ema_momentum)
            return self.ema_background.to(dtype=flat.dtype)
        if mode in {"same_region", "region_mean", "mean"}:
            return current.detach()
        if mode == "image_mean":
            return flat[sample_idx].mean(dim=0, keepdim=True).detach()
        if mode == "zero":
            return torch.zeros_like(current)
        raise ValueError(f"Unsupported same_region_background={same_region_background!r}")

    def _replace(
        self,
        patch_tokens: torch.Tensor,
        flat_idx: torch.Tensor,
        background_idx: torch.Tensor | None = None,
        same_region_background: str = "ema",
    ) -> torch.Tensor:
        b, layers, n, d = patch_tokens.shape
        patched = patch_tokens.clone()
        flat = patched.reshape(b, layers * n, d)
        for i in range(b):
            idx = flat_idx[i].unique().clamp(0, layers * n - 1)
            if background_idx is None:
                bg_pool = idx
            else:
                bg_pool = background_idx[i].unique().clamp(0, layers * n - 1)
            # Same-region replacement: selected and random deletions are filled
            # from equal-area tokens sampled from the same factor region. The
            # default EMA mode makes the configured background stateful instead
            # of silently falling back to a one-off region mean.
            bg = self._background(flat, i, bg_pool, same_region_background)
            flat[i, idx] = bg
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
        random_indices: torch.Tensor | None = None,
    ) -> dict:
        b, factors, k = topk_indices.shape
        targets = credit_norm.shape[-1]
        device = patch_tokens.device
        selected_effect = torch.zeros(b, targets, device=device, dtype=patch_tokens.dtype)
        random_effect = torch.zeros_like(selected_effect)
        selected_effect_all_targets = torch.zeros(b, targets, targets, device=device, dtype=patch_tokens.dtype)
        random_effect_all_targets = torch.zeros_like(selected_effect_all_targets)
        credit_sign = torch.zeros_like(selected_effect)
        selected_credit_value = torch.zeros_like(selected_effect)
        selected_factor_id = torch.full((b, targets), -1, device=device, dtype=torch.long)
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
                if random_indices is not None:
                    rand_idx = random_indices[i, f].view(1, -1)
                else:
                    rand_idx = topk_indices[i, f, torch.randperm(topk_indices.shape[-1], device=device)].view(1, -1)
                patched_sel = self._replace(
                    patch_tokens[i : i + 1],
                    sel_idx,
                    rand_idx,
                    same_region_background=same_region_background,
                )
                patched_rand = self._replace(
                    patch_tokens[i : i + 1],
                    rand_idx,
                    sel_idx,
                    same_region_background=same_region_background,
                )
                sel_logits = head_fn(patched_sel)
                rnd_logits = head_fn(patched_rand)
                selected_delta_all = target_logits[i] - sel_logits[0]
                random_delta_all = target_logits[i] - rnd_logits[0]
                selected_effect[i, t] = target_logits[i, t] - sel_logits[0, t]
                random_effect[i, t] = target_logits[i, t] - rnd_logits[0, t]
                selected_effect_all_targets[i, t] = selected_delta_all
                random_effect_all_targets[i, t] = random_delta_all
                credit_sign[i, t] = credit_norm[i, f, t].sign()
                selected_credit_value[i, t] = credit_norm[i, f, t]
                selected_factor_id[i, t] = f
                selected_mask[i, t] = True
                used += 1
        raw_gap = selected_effect - random_effect
        # Positive-credit factors should make z_t drop when deleted; inhibitory
        # negative-credit factors should make z_t rise when deleted. Use the
        # credit sign so both support and inhibit evidence can pass the same
        # selected-vs-random gate instead of silently dropping all inhibitors.
        gap = raw_gap * credit_sign.where(credit_sign != 0, torch.ones_like(credit_sign))
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
            "selected_effect_all_targets": selected_effect_all_targets,
            "random_effect_all_targets": random_effect_all_targets,
            "raw_selected_vs_random_gap": raw_gap,
            "credit_sign": credit_sign,
            "selected_credit_value": selected_credit_value,
            "selected_factor_id": selected_factor_id,
            "selected_pair_mask": selected_mask,
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
