from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SAVE_REASON_LOSS_WEIGHTS = {
    "benchmark": 1.00,
    "private_direct": 0.35,
    "clean": 0.35,
    "rank": 0.06,
    "soft_f1": 0.03,
    "bbam": 0.03,
    "view_consistency": 0.02,
    "pu_private": 1.00,
}


def _tensor(output: Mapping[str, Any], *names: str) -> Tensor:
    for name in names:
        value = output.get(name)
        if isinstance(value, Tensor):
            return value
    raise KeyError(f"output is missing all of {names}")


def _optional(output: Mapping[str, Any], anchor: Tensor, *names: str) -> Tensor | None:
    for name in names:
        value = output.get(name)
        if value is not None:
            return value if isinstance(value, Tensor) else anchor.new_tensor(value)
    return None


def asymmetric_reason_elements(
    logits: Tensor,
    target: Tensor,
    *,
    gamma_negative: float = 2.0,
) -> tuple[Tensor, Tensor]:
    if logits.shape != target.shape:
        raise ValueError("reason logits and target must have the same shape")
    probability = torch.sigmoid(logits.float())
    target = target.to(probability).clamp(0.0, 1.0)
    positive = -target * torch.log(probability.clamp_min(1e-6))
    negative = -(
        (1.0 - target)
        * probability.pow(float(gamma_negative))
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )
    return positive, negative


def weighted_clean_reason_asl(
    logits: Tensor,
    target: Tensor,
    reliability: Tensor,
    *,
    certified_negative: Tensor | None = None,
) -> Tensor:
    """Clean supervision weights are confidence-only and never train confidence."""
    positive, negative = asymmetric_reason_elements(logits, target)
    q = reliability.detach().to(logits).clamp(0.0, 1.0)
    positive_weight = 0.50 + 0.50 * q
    negative_weight = q
    if certified_negative is not None:
        certified = certified_negative.to(logits).bool()
        negative_weight = torch.where(certified, torch.ones_like(negative_weight), negative_weight)
    target = target.to(logits)
    return (
        positive * positive_weight * target
        + negative * negative_weight * (1.0 - target)
    ).mean()


def reason_soft_f1_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.to(probability)
    tp = (probability * target).sum(0)
    fp = (probability * (1.0 - target)).sum(0)
    fn = ((1.0 - probability) * target).sum(0)
    return 1.0 - ((2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)).mean()


def reason_rank_loss(
    logits: Tensor,
    target: Tensor,
    reliability: Tensor,
    *,
    margin: float = 0.20,
) -> Tensor:
    if logits.shape != target.shape:
        raise ValueError("reason logits and target must have the same shape")
    q = reliability.detach().to(logits).clamp(0.0, 1.0)
    positive = target.to(logits).gt(0.5)
    negative = ~positive
    positive_weight = positive.to(logits) * (0.50 + 0.50 * q)
    negative_weight = negative.to(logits) * q
    pair_weight = positive_weight[:, None, :] * negative_weight[None, :, :]
    pair_count = pair_weight.sum()
    if not bool(pair_count.gt(0)):
        return logits.new_zeros(())
    pair_margin = float(margin) - logits[:, None, :] + logits[None, :, :]
    return (torch.relu(pair_margin) * pair_weight).sum() / pair_count.clamp_min(1e-6)


def select_tail_reason_ids(train_main_targets: Tensor, tail_count: int = 8) -> Tensor:
    """Select rare labels from train-main only; no test statistics enter this API."""
    if train_main_targets.ndim != 2:
        raise ValueError("train_main_targets must have shape [B,reason_dim]")
    if tail_count <= 0:
        return torch.empty(0, dtype=torch.long, device=train_main_targets.device)
    frequency = train_main_targets.detach().float().gt(0.5).sum(0)
    count = min(int(tail_count), int(frequency.numel()))
    return torch.argsort(frequency, descending=False)[:count]


def _normalized(value: Tensor, eps: float = 1e-6) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=eps)


def balanced_angular_margin_loss(
    embeddings: Tensor,
    target: Tensor,
    *,
    tail_reason_ids: Tensor | Sequence[int],
    positive_prototypes: Tensor | None = None,
    negative_prototypes: Tensor | None = None,
    margin: float = 0.15,
) -> Tensor:
    """Apply low-weight balanced angular margins only to train-main tail labels."""
    if embeddings.ndim != 3 or target.ndim != 2 or embeddings.shape[:2] != target.shape:
        raise ValueError("embeddings must be [B,R,D] and target must be [B,R]")
    ids = torch.as_tensor(tail_reason_ids, device=embeddings.device, dtype=torch.long).flatten()
    ids = ids[(ids >= 0) & (ids < embeddings.shape[1])]
    if ids.numel() == 0:
        return embeddings.new_zeros(())
    normalized = _normalized(embeddings)
    if positive_prototypes is None or negative_prototypes is None:
        pos_proto = []
        neg_proto = []
        for reason_id in ids.tolist():
            positive = target[:, reason_id].gt(0.5)
            negative = ~positive
            positive_mean = (
                normalized[positive, reason_id].mean(0)
                if bool(positive.any())
                else None
            )
            negative_mean = (
                normalized[negative, reason_id].mean(0)
                if bool(negative.any())
                else None
            )
            if positive_mean is None:
                assert negative_mean is not None
                positive_mean = -negative_mean
            if negative_mean is None:
                assert positive_mean is not None
                negative_mean = -positive_mean
            pos_proto.append(positive_mean)
            neg_proto.append(negative_mean)
        positive_prototypes = torch.stack(pos_proto, dim=0).detach()
        negative_prototypes = torch.stack(neg_proto, dim=0).detach()
    else:
        positive_prototypes = positive_prototypes.to(embeddings).detach()
        negative_prototypes = negative_prototypes.to(embeddings).detach()
        if positive_prototypes.ndim != 2 or negative_prototypes.shape != positive_prototypes.shape:
            raise ValueError("prototype tensors must have shape [R,D]")
        positive_prototypes = positive_prototypes[ids]
        negative_prototypes = negative_prototypes[ids]
    positive_prototypes = _normalized(positive_prototypes).view(1, -1, embeddings.shape[-1])
    negative_prototypes = _normalized(negative_prototypes).view(1, -1, embeddings.shape[-1])
    selected = normalized[:, ids]
    positive_cosine = (selected * positive_prototypes).sum(-1)
    negative_cosine = (selected * negative_prototypes).sum(-1)
    selected_target = target[:, ids].to(embeddings)
    direction = selected_target * 2.0 - 1.0
    return F.relu(float(margin) - direction * (positive_cosine - negative_cosine)).mean()


class SAVETailReasonGeometry(nn.Module):
    """EMA positive/negative prototypes for the train-main tail labels."""

    def __init__(
        self,
        reason_dim: int = 21,
        embedding_dim: int = 384,
        *,
        momentum: float = 0.95,
        margin: float = 0.15,
    ) -> None:
        super().__init__()
        if not 0.0 <= float(momentum) < 1.0:
            raise ValueError("momentum must lie in [0,1)")
        self.momentum = float(momentum)
        self.margin = float(margin)
        self.register_buffer("positive_prototypes", torch.zeros(reason_dim, embedding_dim))
        self.register_buffer("negative_prototypes", torch.zeros(reason_dim, embedding_dim))
        self.register_buffer("prototype_updates", torch.zeros(reason_dim, dtype=torch.long))

    @torch.no_grad()
    def update(self, embeddings: Tensor, target: Tensor, tail_reason_ids: Tensor | Sequence[int]) -> None:
        normalized = _normalized(embeddings.detach())
        ids = torch.as_tensor(tail_reason_ids, device=embeddings.device, dtype=torch.long).flatten()
        for reason_id in ids.tolist():
            if reason_id < 0 or reason_id >= normalized.shape[1]:
                continue
            positive = target[:, reason_id].gt(0.5)
            negative = ~positive
            for mask, buffer in ((positive, self.positive_prototypes), (negative, self.negative_prototypes)):
                if not bool(mask.any()):
                    continue
                mean = _normalized(normalized[mask, reason_id].mean(0, keepdim=True)).squeeze(0)
                if int(self.prototype_updates[reason_id]) == 0:
                    buffer[reason_id].copy_(mean.to(buffer))
                else:
                    buffer[reason_id].mul_(self.momentum).add_(mean.to(buffer) * (1.0 - self.momentum))
                    buffer[reason_id].copy_(_normalized(buffer[reason_id].view(1, -1)).squeeze(0))
            self.prototype_updates[reason_id].add_(1)

    def forward(self, embeddings: Tensor, target: Tensor, tail_reason_ids: Tensor | Sequence[int]) -> Tensor:
        return balanced_angular_margin_loss(
            embeddings,
            target,
            tail_reason_ids=tail_reason_ids,
            positive_prototypes=self.positive_prototypes,
            negative_prototypes=self.negative_prototypes,
            margin=self.margin,
        )


def reason_view_consistency_loss(
    logits: Tensor,
    view_logits: Tensor,
    reliability: Tensor,
    view_reliability: Tensor | None = None,
) -> Tensor:
    if logits.shape != view_logits.shape:
        raise ValueError("view reason logits must have the same shape")
    q = reliability.detach().to(logits).clamp(0.0, 1.0)
    if view_reliability is not None:
        q = q * view_reliability.detach().to(logits).clamp(0.0, 1.0)
    difference = (torch.sigmoid(logits.float()) - torch.sigmoid(view_logits.float())).square()
    return (difference * q).sum() / q.sum().clamp_min(1e-6)


def private_pu_reason_loss(
    logits: Tensor,
    target: Tensor,
    reliability: Tensor,
    *,
    positive_state_probability: Tensor | None = None,
    pu_lambda: Tensor | None = None,
) -> Tensor:
    q = reliability.detach().to(logits).clamp(0.0, 1.0)
    if positive_state_probability is not None:
        q = q * positive_state_probability.detach().to(logits).clamp(0.0, 1.0)
    if pu_lambda is not None:
        q = q * pu_lambda.detach().to(logits).clamp(0.0, 0.10)
    positive, negative = asymmetric_reason_elements(logits, target)
    target = target.to(logits)
    return (positive * target + negative * q * (1.0 - target)).mean()


def save_reason_loss(
    output: Mapping[str, Any],
    target: Tensor,
    *,
    reliability: Tensor | None = None,
    certified_negative: Tensor | None = None,
    tail_reason_ids: Tensor | Sequence[int] | None = None,
    positive_prototypes: Tensor | None = None,
    negative_prototypes: Tensor | None = None,
    view_output: Mapping[str, Any] | None = None,
    pu_lambda: Tensor | None = None,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Tensor]:
    """Single-owner reason loss implementing the exact section 16.2 composition."""
    benchmark = _tensor(output, "reason_logits_benchmark", "reason_logits_bench", "reason_logits_final")
    private_direct = _tensor(
        output,
        "reason_logits_private_direct",
        "reason_logits_private-direct",
        "reason_logits_private",
    )
    clean = _tensor(output, "reason_logits_clean", "reason_logits_shared")
    if target.shape != benchmark.shape or private_direct.shape != target.shape or clean.shape != target.shape:
        raise ValueError("all reason branches must match the reason target shape")
    target = target.to(benchmark)
    if reliability is None:
        reliability = _optional(output, benchmark, "reason_reliability", "factor_reliability")
    if reliability is None:
        reliability = benchmark.new_ones(benchmark.shape)
    if reliability.shape != target.shape:
        raise ValueError("reason reliability must match the reason target shape")

    positive_benchmark, negative_benchmark = asymmetric_reason_elements(benchmark, target)
    benchmark_loss = (positive_benchmark + negative_benchmark).mean()
    positive_direct, negative_direct = asymmetric_reason_elements(private_direct, target)
    private_direct_loss = (positive_direct + negative_direct).mean()
    clean_loss = weighted_clean_reason_asl(
        clean,
        target,
        reliability,
        certified_negative=certified_negative,
    )
    rank = _optional(output, benchmark, "reason_rank_loss", "loss_rank")
    if rank is None:
        rank = reason_rank_loss(benchmark, target, reliability)
    else:
        rank = rank.float().mean()
    soft_f1 = _optional(output, benchmark, "reason_soft_f1_loss", "loss_soft_f1")
    if soft_f1 is None:
        soft_f1 = reason_soft_f1_loss(benchmark, target)
    else:
        soft_f1 = soft_f1.float().mean()

    bbam = _optional(output, benchmark, "reason_bbam_loss", "loss_bbam")
    if bbam is None and tail_reason_ids is not None:
        embedding = _optional(output, benchmark, "reason_embedding_private", "private_reason_embedding")
        if embedding is not None:
            bbam = balanced_angular_margin_loss(
                embedding,
                target,
                tail_reason_ids=tail_reason_ids,
                positive_prototypes=positive_prototypes,
                negative_prototypes=negative_prototypes,
            )
    if bbam is None:
        bbam = benchmark.new_zeros(())
    else:
        bbam = bbam.float().mean()

    view = _optional(output, benchmark, "reason_view_consistency_loss", "loss_view_consistency")
    if view is None and view_output is not None:
        view_logits = _tensor(view_output, "reason_logits_benchmark", "reason_logits_final")
        view_reliability = _optional(view_output, benchmark, "reason_reliability", "factor_reliability")
        view = reason_view_consistency_loss(benchmark, view_logits, reliability, view_reliability)
    if view is None:
        view = benchmark.new_zeros(())
    else:
        view = view.float().mean()

    pu = _optional(output, benchmark, "reason_pu_private_loss", "loss_pu_private")
    if pu is None:
        pu_logits = _optional(output, benchmark, "reason_logits_pu_private")
        if pu_logits is not None:
            pu = private_pu_reason_loss(
                pu_logits, target, reliability, pu_lambda=pu_lambda
            )
    if pu is None:
        pu = benchmark.new_zeros(())
    else:
        pu = pu.float().mean()

    selected = dict(SAVE_REASON_LOSS_WEIGHTS)
    if weights is not None:
        selected.update({key: float(value) for key, value in weights.items()})
    total = (
        selected["benchmark"] * benchmark_loss
        + selected["private_direct"] * private_direct_loss
        + selected["clean"] * clean_loss
        + selected["rank"] * rank
        + selected["soft_f1"] * soft_f1
        + selected["bbam"] * bbam
        + selected["view_consistency"] * view
        + selected["pu_private"] * pu
    )
    return {
        "benchmark": benchmark_loss,
        "private_direct": private_direct_loss,
        "clean": clean_loss,
        "rank": rank,
        "soft_f1": soft_f1,
        "bbam": bbam,
        "view_consistency": view,
        "pu_private": pu,
        "total": total,
    }


build_save_reason_loss = save_reason_loss
save_reason_losses = save_reason_loss
save_reason_loss_bundle = save_reason_loss


__all__ = [
    "SAVE_REASON_LOSS_WEIGHTS",
    "SAVETailReasonGeometry",
    "asymmetric_reason_elements",
    "balanced_angular_margin_loss",
    "build_save_reason_loss",
    "private_pu_reason_loss",
    "reason_rank_loss",
    "reason_soft_f1_loss",
    "reason_view_consistency_loss",
    "save_reason_loss",
    "save_reason_loss_bundle",
    "save_reason_losses",
    "select_tail_reason_ids",
    "weighted_clean_reason_asl",
]
