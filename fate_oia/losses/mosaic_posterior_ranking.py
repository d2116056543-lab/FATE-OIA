from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue


def _cross_image_ranking_loss(
    current_logits: torch.Tensor,
    current_positive_weight: torch.Tensor,
    sample_ids: Sequence[str | int],
    queue: MOSAICSoftRankQueue,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if tuple(current_logits.shape) != tuple(current_positive_weight.shape):
        raise ValueError("ranking logits and weights must have matching shapes")
    if current_logits.shape[1] != queue.label_dim or len(sample_ids) != current_logits.shape[0]:
        raise ValueError("ranking batch does not match queue label/sample contract")
    history = queue.snapshot()
    if history["logits"].shape[0] == 0:
        zero = current_logits.sum() * 0.0
        return zero, {"pair_weight_sum": zero.detach(), "queue_count": zero.new_tensor(0)}

    history_logits = history["logits"].to(device=current_logits.device, dtype=current_logits.dtype)
    history_targets = history["targets"].to(device=current_logits.device, dtype=current_logits.dtype)
    history_hashes = history["sample_hashes"].to(device=current_logits.device)
    current_hashes = queue.hash_sample_ids(sample_ids, device=current_logits.device)
    cross_sample = current_hashes[:, None] != history_hashes[None, :]
    pair_weights = (
        current_positive_weight.detach()[:, None, :]
        * (1.0 - history_targets.detach()[None, :, :])
        * cross_sample[:, :, None].to(dtype=current_logits.dtype)
    )
    margin_penalty = F.softplus(-(current_logits[:, None, :] - history_logits[None, :, :]))
    weight_sum = pair_weights.sum()
    loss = (pair_weights * margin_penalty).sum() / weight_sum.clamp_min(1e-12)
    loss = torch.where(weight_sum > 0, loss, current_logits.sum() * 0.0)
    return loss, {
        "pair_weight_sum": weight_sum.detach(),
        "queue_count": weight_sum.new_tensor(history_logits.shape[0]).detach(),
    }


def posterior_weighted_reason_ranking_loss(
    current_logits: torch.Tensor,
    current_posterior: torch.Tensor,
    sample_ids: Sequence[str | int],
    queue: MOSAICSoftRankQueue,
    *,
    label_gate: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if label_gate is not None:
        if label_gate.shape != (current_logits.shape[1],) or label_gate.dtype != torch.bool:
            raise ValueError("posterior reason ranking label gate must be bool [R]")
        # Admission is per label. A disabled label contributes neither a
        # current positive nor a queue comparison to PU ranking.
        current_posterior = current_posterior * label_gate.to(
            dtype=current_posterior.dtype, device=current_posterior.device
        ).view(1, -1)
    return _cross_image_ranking_loss(current_logits, current_posterior, sample_ids, queue)


def observed_reason_cross_image_ranking_loss(
    current_logits: torch.Tensor,
    observed_targets: torch.Tensor,
    observed_valid_mask: torch.Tensor,
    sample_ids: Sequence[str | int],
    queue: MOSAICSoftRankQueue,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Rank known observed positives above known cross-image negatives only.

    Synthetic hidden positives are represented by ``False`` in
    ``observed_valid_mask``.  The queue persists them as NaN sentinels, so
    they can never be reintroduced as hard negatives on later updates.
    """
    if (
        current_logits.shape != observed_targets.shape
        or current_logits.shape != observed_valid_mask.shape
        or observed_valid_mask.dtype != torch.bool
    ):
        raise ValueError("observed reason ranking requires matching logits, targets, and bool valid mask")
    if current_logits.shape[1] != queue.label_dim or len(sample_ids) != current_logits.shape[0]:
        raise ValueError("observed reason ranking batch does not match queue label/sample contract")
    history = queue.snapshot()
    if history["logits"].shape[0] == 0:
        zero = current_logits.sum() * 0.0
        return zero, {"pair_weight_sum": zero.detach(), "queue_count": zero.new_tensor(0)}

    history_logits = history["logits"].to(device=current_logits.device, dtype=current_logits.dtype)
    history_targets = history["targets"].to(device=current_logits.device, dtype=current_logits.dtype)
    history_hashes = history["sample_hashes"].to(device=current_logits.device)
    current_hashes = queue.hash_sample_ids(sample_ids, device=current_logits.device)
    current_positive = observed_targets.to(dtype=current_logits.dtype) * observed_valid_mask.to(dtype=current_logits.dtype)
    history_known = torch.isfinite(history_targets)
    history_negative = (1.0 - history_targets.nan_to_num(0.0)).clamp(0.0, 1.0) * history_known.to(dtype=current_logits.dtype)
    cross_sample = current_hashes[:, None] != history_hashes[None, :]
    pair_weights = (
        current_positive.detach()[:, None, :]
        * history_negative.detach()[None, :, :]
        * cross_sample[:, :, None].to(dtype=current_logits.dtype)
    )
    margin_penalty = F.softplus(-(current_logits[:, None, :] - history_logits[None, :, :]))
    weight_sum = pair_weights.sum()
    loss = (pair_weights * margin_penalty).sum() / weight_sum.clamp_min(1e-12)
    loss = torch.where(weight_sum > 0, loss, current_logits.sum() * 0.0)
    return loss, {
        "pair_weight_sum": weight_sum.detach(),
        "queue_count": weight_sum.new_tensor(history_logits.shape[0]).detach(),
    }


def action_cross_image_ranking_loss(
    current_logits: torch.Tensor,
    action_targets: torch.Tensor,
    sample_ids: Sequence[str | int],
    queue: MOSAICSoftRankQueue,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _cross_image_ranking_loss(current_logits, action_targets, sample_ids, queue)
