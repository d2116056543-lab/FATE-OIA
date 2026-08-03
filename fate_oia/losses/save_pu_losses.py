from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_BBAM_TAIL_COUNT = 8
DEFAULT_BBAM_MARGIN = 0.15
DEFAULT_BBAM_PROTOTYPE_MOMENTUM = 0.95
DEFAULT_PU_LCB_SCALE = 0.05
DEFAULT_PU_LAMBDA_MAX = 0.10
DEFAULT_VIEW_MOMENTUM = 0.95
DEFAULT_VIEW_ALPHA = 1.0
DEFAULT_VIEW_TEMPERATURE = 1.0
SAVE_TAIL_REASON_ARTIFACT = "artifacts/save/tail_reason_ids.json"

_TRAIN_SPLITS = frozenset({"train", "train_main", "train_audit", "train_calib"})


def _canonical_split_name(split_name: str) -> str:
    if not isinstance(split_name, str) or not split_name.strip():
        raise ValueError("split_name must be a non-empty string")
    return split_name.strip().lower().replace("-", "_")


def _require_split(split_name: str, expected: str) -> str:
    actual = _canonical_split_name(split_name)
    expected = _canonical_split_name(expected)
    if actual != expected:
        raise ValueError(
            f"SAVE provenance requires {expected}; received {actual}"
        )
    return actual


def _require_train_split(split_name: str) -> str:
    actual = _canonical_split_name(split_name)
    if actual not in _TRAIN_SPLITS:
        raise ValueError(
            "SAVE view consistency state is train-only; "
            f"received non-training split {actual}"
        )
    return actual


def _reason_matrix(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, reason_dim]")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one sample and label")
    return value


def _reason_embeddings(value: Tensor, name: str = "private_embeddings") -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, reason_dim, embedding_dim]")
    if value.shape[0] == 0 or value.shape[1] == 0 or value.shape[2] == 0:
        raise ValueError(f"{name} must contain non-empty dimensions")
    return value


def _tail_ids(tail_reason_ids: Sequence[int] | Tensor, reason_dim: int) -> list[int]:
    if isinstance(tail_reason_ids, Tensor):
        values = tail_reason_ids.detach().cpu().view(-1).tolist()
    else:
        values = list(tail_reason_ids)
    ids = [int(value) for value in values]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("tail_reason_ids must contain unique label ids")
    if any(value < 0 or value >= reason_dim for value in ids):
        raise ValueError("tail_reason_ids contains an out-of-range label")
    return ids


def select_tail_reason_ids(
    reason_targets: Tensor,
    *,
    split_name: str = "train_main",
    tail_count: int = DEFAULT_BBAM_TAIL_COUNT,
) -> list[int]:
    """Select the least frequent reason labels from train-main only.

    Ties are resolved by the stable label id, so the resulting artifact is
    reproducible without consulting evaluation metrics or a test split.
    """
    _require_split(split_name, "train_main")
    targets = _reason_matrix(reason_targets, "reason_targets").detach()
    if int(tail_count) <= 0 or int(tail_count) > targets.shape[1]:
        raise ValueError("tail_count must be in [1, reason_dim]")
    counts = targets.gt(0.5).sum(dim=0).cpu().tolist()
    ordered = sorted(range(targets.shape[1]), key=lambda label: (counts[label], label))
    return ordered[: int(tail_count)]


def build_bbam_tail_spec(
    train_main_targets: Tensor,
    *,
    split_name: str = "train_main",
    tail_count: int = DEFAULT_BBAM_TAIL_COUNT,
) -> dict[str, Any]:
    """Build the auditable BBAM tail selection contract."""
    _require_split(split_name, "train_main")
    targets = _reason_matrix(train_main_targets, "train_main_targets").detach()
    tail_ids = select_tail_reason_ids(
        targets,
        split_name="train_main",
        tail_count=tail_count,
    )
    return {
        "source_split": "train_main",
        "selection": "bottom_reason_frequency",
        "tail_count": int(tail_count),
        "tail_reason_ids": tail_ids,
        "positive_count": [
            int(value) for value in targets.gt(0.5).sum(dim=0).cpu().tolist()
        ],
        "artifact": SAVE_TAIL_REASON_ARTIFACT,
        "test_metrics_used": False,
    }


def _normalise(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8)


def _selected_prototypes(
    prototypes: Tensor,
    *,
    tail_ids: list[int],
    reason_dim: int,
    embedding_dim: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    if not isinstance(prototypes, Tensor) or prototypes.ndim != 2:
        raise ValueError(f"{name} must have shape [reason_dim, embedding_dim]")
    if prototypes.shape[1] != embedding_dim:
        raise ValueError(f"{name} embedding dimension does not match private embeddings")
    if prototypes.shape[0] == reason_dim:
        selected = prototypes[tail_ids]
    elif prototypes.shape[0] == len(tail_ids):
        selected = prototypes
    else:
        raise ValueError(f"{name} must contain all reason labels or only tail labels")
    return selected.detach().to(device=reference.device, dtype=torch.float32)


def balanced_angular_margin_loss(
    private_embeddings: Tensor,
    reason_targets: Tensor,
    tail_reason_ids: Sequence[int] | Tensor,
    *,
    positive_prototypes: Tensor | None = None,
    negative_prototypes: Tensor | None = None,
    split_name: str = "train_main",
    margin: float = DEFAULT_BBAM_MARGIN,
) -> Tensor:
    """Compute BBAM on the train-main-selected reason tail only.

    Each tail label is balanced between observed positives and negatives.  The
    negative side is represented by a same-label prototype, optionally built
    from matched low-overlap background by ``BBAMPrototypeBank``.  No pair
    memory or non-tail label contributes to this loss.
    """
    _require_split(split_name, "train_main")
    embeddings = _reason_embeddings(private_embeddings)
    targets = _reason_matrix(reason_targets, "reason_targets")
    if targets.shape != embeddings.shape[:2]:
        raise ValueError("reason_targets must match private_embeddings batch and labels")
    if float(margin) <= 0.0:
        raise ValueError("margin must be positive")
    batch, reason_dim, embedding_dim = embeddings.shape
    tail_ids = _tail_ids(tail_reason_ids, reason_dim)
    if positive_prototypes is None or negative_prototypes is None:
        raise ValueError(
            "BBAM requires initialized positive and negative prototypes"
        )
    hidden = _normalise(embeddings)
    positive = _selected_prototypes(
        positive_prototypes,
        tail_ids=tail_ids,
        reason_dim=reason_dim,
        embedding_dim=embedding_dim,
        reference=hidden,
        name="positive_prototypes",
    )
    negative = _selected_prototypes(
        negative_prototypes,
        tail_ids=tail_ids,
        reason_dim=reason_dim,
        embedding_dim=embedding_dim,
        reference=hidden,
        name="negative_prototypes",
    )
    positive = _normalise(positive)
    negative = _normalise(negative)
    identical = torch.isclose(positive, negative, rtol=1e-5, atol=1e-6).all(dim=-1)
    if bool(identical.any().item()):
        raise ValueError("BBAM positive and negative prototypes must be distinct")

    tail_hidden = hidden[:, tail_ids]
    target = targets[:, tail_ids].detach().to(device=hidden.device).gt(0.5)
    cos_positive = (tail_hidden * positive.view(1, len(tail_ids), embedding_dim)).sum(-1)
    cos_negative = (tail_hidden * negative.view(1, len(tail_ids), embedding_dim)).sum(-1)
    positive_term = F.relu(float(margin) - cos_positive + cos_negative)
    negative_term = F.relu(float(margin) - cos_negative + cos_positive)

    label_terms: list[Tensor] = []
    for index in range(len(tail_ids)):
        terms: list[Tensor] = []
        positive_mask = target[:, index]
        negative_mask = ~positive_mask
        if bool(positive_mask.any()):
            terms.append(positive_term[positive_mask, index].mean())
        if bool(negative_mask.any()):
            terms.append(negative_term[negative_mask, index].mean())
        if terms:
            label_terms.append(torch.stack(terms).mean())
    if not label_terms:
        return embeddings.sum() * 0.0
    return torch.stack(label_terms).mean()


class BBAMPrototypeBank(nn.Module):
    """EMA positive/negative prototypes owned by train-main BBAM."""

    def __init__(
        self,
        *,
        reason_dim: int = 21,
        embedding_dim: int = 384,
        tail_reason_ids: Sequence[int] | Tensor | None = None,
        momentum: float = DEFAULT_BBAM_PROTOTYPE_MOMENTUM,
    ) -> None:
        super().__init__()
        if int(reason_dim) <= 0 or int(embedding_dim) <= 0:
            raise ValueError("reason_dim and embedding_dim must be positive")
        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("prototype momentum must be in (0, 1)")
        if tail_reason_ids is None:
            tail_reason_ids = list(range(min(DEFAULT_BBAM_TAIL_COUNT, int(reason_dim))))
        self.reason_dim = int(reason_dim)
        self.embedding_dim = int(embedding_dim)
        self.tail_reason_ids = tuple(_tail_ids(tail_reason_ids, self.reason_dim))
        self.momentum = float(momentum)
        self.register_buffer(
            "positive_prototypes",
            torch.zeros(self.reason_dim, self.embedding_dim),
            persistent=True,
        )
        self.register_buffer(
            "negative_prototypes",
            torch.zeros(self.reason_dim, self.embedding_dim),
            persistent=True,
        )
        self.register_buffer(
            "positive_updates", torch.zeros(self.reason_dim, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "negative_updates", torch.zeros(self.reason_dim, dtype=torch.long), persistent=True
        )

    def _update_one(self, current: Tensor, value: Tensor, initialized: bool) -> Tensor:
        value = _normalise(value.view(1, -1)).view(-1)
        if not initialized:
            return value.to(device=current.device, dtype=current.dtype)
        return _normalise(
            self.momentum * current + (1.0 - self.momentum) * value.to(current)
        ).to(dtype=current.dtype)

    def update(
        self,
        private_embeddings: Tensor,
        reason_targets: Tensor,
        *,
        split_name: str = "train_main",
        split: str | None = None,
        tail_reason_ids: Sequence[int] | Tensor | None = None,
        negative_embeddings: Tensor | None = None,
        matched_background_embeddings: Tensor | None = None,
        negative_confidence: Tensor | None = None,
        min_negative_confidence: float = 0.5,
        tail_spec: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update prototypes from detached train-main representations only."""
        if split is not None:
            split_name = split
        _require_split(split_name, "train_main")
        if tail_spec is not None:
            if tail_spec.get("source_split") != "train_main":
                raise ValueError("BBAM tail spec must come from train_main")
            tail_reason_ids = tail_spec.get("tail_reason_ids")
        if tail_reason_ids is None:
            tail_ids = list(self.tail_reason_ids)
        else:
            tail_ids = _tail_ids(tail_reason_ids, self.reason_dim)
        embeddings = _reason_embeddings(private_embeddings)
        targets = _reason_matrix(reason_targets, "reason_targets")
        if tuple(embeddings.shape[:2]) != tuple(targets.shape):
            raise ValueError("reason_targets must match private_embeddings")
        if embeddings.shape[1] != self.reason_dim or embeddings.shape[2] != self.embedding_dim:
            raise ValueError("prototype bank dimensions do not match private_embeddings")

        background = matched_background_embeddings
        if background is None:
            background = negative_embeddings
        if background is not None:
            if not isinstance(background, Tensor) or background.ndim not in (2, 3):
                raise ValueError("negative/background embeddings must be [B,D] or [B,R,D]")
            if background.shape[0] != embeddings.shape[0] or background.shape[-1] != self.embedding_dim:
                raise ValueError("negative/background embeddings have incompatible shape")
            if background.ndim == 3 and background.shape[1] != self.reason_dim:
                raise ValueError("per-label background embeddings must match reason_dim")
        confidence = None
        if negative_confidence is not None:
            confidence = _reason_matrix(negative_confidence, "negative_confidence")
            if confidence.shape != targets.shape:
                raise ValueError("negative_confidence must match reason_targets")

        updated_positive: list[int] = []
        updated_negative: list[int] = []
        with torch.no_grad():
            hidden = _normalise(embeddings.detach())
            for label in tail_ids:
                positive_mask = targets[:, label].detach().to(hidden.device).gt(0.5)
                if bool(positive_mask.any()):
                    value = hidden[positive_mask, label].mean(0)
                    initialized = bool(self.positive_updates[label].item())
                    self.positive_prototypes[label].copy_(
                        self._update_one(self.positive_prototypes[label], value, initialized)
                    )
                    self.positive_updates[label].add_(1)
                    updated_positive.append(label)

                negative_mask = ~positive_mask
                if confidence is not None:
                    negative_mask &= confidence[:, label].detach().to(hidden.device) >= float(
                        min_negative_confidence
                    )
                if not bool(negative_mask.any()):
                    continue
                if background is None:
                    value = hidden[negative_mask, label].mean(0)
                elif background.ndim == 3:
                    value = _normalise(background.detach().to(hidden.device))[
                        negative_mask, label
                    ].mean(0)
                else:
                    value = _normalise(background.detach().to(hidden.device))[
                        negative_mask
                    ].mean(0)
                initialized = bool(self.negative_updates[label].item())
                self.negative_prototypes[label].copy_(
                    self._update_one(self.negative_prototypes[label], value, initialized)
                )
                self.negative_updates[label].add_(1)
                updated_negative.append(label)
        return {
            "source_split": "train_main",
            "tail_reason_ids": list(tail_ids),
            "updated_positive": updated_positive,
            "updated_negative": updated_negative,
        }


def save_bbam_loss(
    private_embeddings: Tensor,
    reason_targets: Tensor,
    *,
    tail_reason_ids: Sequence[int] | Tensor | None = None,
    tail_spec: Mapping[str, Any] | None = None,
    prototype_bank: BBAMPrototypeBank | None = None,
    split_name: str = "train_main",
    margin: float = DEFAULT_BBAM_MARGIN,
) -> Tensor:
    """Return raw BBAM; the SAVE loss registry owns its sole weight."""
    _require_split(split_name, "train_main")
    if tail_spec is not None:
        if tail_spec.get("source_split") != "train_main":
            raise ValueError("BBAM tail spec must come from train_main")
        tail_reason_ids = tail_spec.get("tail_reason_ids")
    if tail_reason_ids is None and prototype_bank is not None:
        tail_reason_ids = prototype_bank.tail_reason_ids
    if tail_reason_ids is None:
        raise ValueError("BBAM requires a train-main tail_reason_ids or tail_spec")
    if prototype_bank is None:
        raise ValueError("BBAM requires an initialized prototype bank")
    tail_ids = _tail_ids(tail_reason_ids, private_embeddings.shape[1])
    bank_indices = torch.tensor(
        tail_ids,
        device=prototype_bank.positive_updates.device,
        dtype=torch.long,
    )
    positive_ready = prototype_bank.positive_updates.index_select(0, bank_indices).gt(0)
    negative_ready = prototype_bank.negative_updates.index_select(0, bank_indices).gt(0)
    if not bool((positive_ready & negative_ready).all().item()):
        raise ValueError("BBAM prototype bank is not initialized for every tail label")
    return balanced_angular_margin_loss(
        private_embeddings,
        reason_targets,
        tail_ids,
        positive_prototypes=prototype_bank.positive_prototypes,
        negative_prototypes=prototype_bank.negative_prototypes,
        split_name=split_name,
        margin=margin,
    )


bbam_loss = save_bbam_loss
balanced_bbam_loss = balanced_angular_margin_loss


def build_reason_reliability(
    visual_reliability: Tensor,
    view_consistency: Tensor,
    action_evidence_overlap: Tensor,
) -> Tensor:
    """Build detached clean-reason reliability q from train-derived evidence."""
    for name, value in (
        ("visual_reliability", visual_reliability),
        ("view_consistency", view_consistency),
        ("action_evidence_overlap", action_evidence_overlap),
    ):
        _reason_matrix(value, name)
    if not (
        visual_reliability.shape == view_consistency.shape == action_evidence_overlap.shape
    ):
        raise ValueError("reason reliability inputs must have equal shapes")
    return (
        visual_reliability.detach().clamp(0.0, 1.0)
        * view_consistency.detach().clamp(0.0, 1.0)
        * action_evidence_overlap.detach().clamp(0.0, 1.0)
    ).clamp(0.0, 1.0).detach()


def pu_score(
    clean_logits: Tensor,
    positive_state_probability: Tensor,
    reliability: Tensor,
    observability: Tensor | None = None,
) -> Tensor:
    """Return detached ``sigmoid(clean) * state-positive * q`` PU scores."""
    _reason_matrix(clean_logits, "clean_logits")
    _reason_matrix(positive_state_probability, "positive_state_probability")
    _reason_matrix(reliability, "reliability")
    if not (
        clean_logits.shape
        == positive_state_probability.shape
        == reliability.shape
    ):
        raise ValueError("PU score inputs must have equal shapes")
    score = (
        torch.sigmoid(clean_logits.detach())
        * positive_state_probability.detach().clamp(0.0, 1.0)
        * reliability.detach().clamp(0.0, 1.0)
    )
    if observability is not None:
        if observability.shape != clean_logits.shape:
            raise ValueError("observability must match clean_logits")
        score = score * observability.detach().clamp(0.0, 1.0)
    return score.clamp(0.0, 1.0).detach()


save_pu_score = pu_score
meter_pu_score = pu_score


def _average_precision(scores: Tensor, targets: Tensor) -> float:
    from fate_oia.metrics import binary_average_precision

    return binary_average_precision(scores, targets)


def _bootstrap_lcb95(
    scores: Tensor,
    targets: Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
    confidence: float,
) -> float:
    if scores.numel() == 0 or targets.sum() <= 0:
        return float("nan")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    scores = scores.detach().float().cpu()
    targets = targets.detach().float().cpu()
    sample_indices = torch.randint(
        scores.numel(),
        (int(bootstrap_samples), scores.numel()),
        generator=generator,
        device="cpu",
    )
    deltas: list[float] = []
    for indices in sample_indices:
        sample_targets = targets[indices]
        if float(sample_targets.sum()) <= 0.0:
            continue
        sample_ap = _average_precision(scores[indices], sample_targets)
        if math.isfinite(sample_ap):
            deltas.append(sample_ap - float(sample_targets.mean()))
    if not deltas:
        return float("nan")
    quantile = max(0.0, min(1.0, 1.0 - float(confidence)))
    return float(torch.quantile(torch.tensor(deltas), quantile).item())


def admit_pu_from_train_audit(
    pu_scores: Tensor,
    observed_targets: Tensor,
    *,
    split_name: str = "train_audit",
    hidden_fraction: float = 0.20,
    bootstrap_samples: int = 512,
    confidence: float = 0.95,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Admit PU labels from hidden-positive train-audit evidence only.

    The admission score is evaluated against deliberately hidden positives and
    observed zeros.  Lambda is label-wise and continuous:
    ``clip(bootstrap_LCB95(lift) / 0.05, 0, 0.10)``.  No test metric or streak
    state participates in this decision.
    """
    _require_split(split_name, "train_audit")
    scores = _reason_matrix(pu_scores, "pu_scores").detach().float().cpu().clamp(0.0, 1.0)
    targets = _reason_matrix(observed_targets, "observed_targets").detach().float().cpu()
    if scores.shape != targets.shape:
        raise ValueError("pu_scores and observed_targets must have equal shapes")
    if not 0.0 < float(hidden_fraction) <= 1.0:
        raise ValueError("hidden_fraction must be in (0, 1]")
    if int(bootstrap_samples) <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")

    labels: list[dict[str, Any]] = []
    lambdas: list[float] = []
    for label in range(targets.shape[1]):
        positive_indices = torch.where(targets[:, label] > 0.5)[0]
        positive_count = int(positive_indices.numel())
        if positive_count == 0:
            row = {
                "label_id": label,
                "audit_split": "train_audit",
                "positive_count": 0,
                "hidden_positive_count": 0,
                "hidden_positive_auprc": float("nan"),
                "frequency_baseline": float("nan"),
                "auprc_lift": float("nan"),
                "bootstrap_lcb95": float("nan"),
                "eligible": False,
                "lambda": 0.0,
            }
            labels.append(row)
            lambdas.append(0.0)
            continue

        generator = torch.Generator(device="cpu").manual_seed(int(seed) + label)
        order = torch.randperm(positive_count, generator=generator)
        hidden_count = max(1, int(round(positive_count * float(hidden_fraction))))
        if positive_count > 1:
            hidden_count = min(hidden_count, positive_count - 1)
        hidden = positive_indices[order[:hidden_count]]
        audit_mask = targets[:, label] <= 0.5
        audit_mask[hidden] = True
        audit_indices = torch.where(audit_mask)[0]
        audit_targets = torch.zeros(audit_indices.numel(), dtype=torch.float32)
        hidden_lookup = torch.isin(audit_indices, hidden)
        audit_targets[hidden_lookup] = 1.0
        audit_scores = scores[audit_indices, label]
        hidden_ap = _average_precision(audit_scores, audit_targets)
        frequency_baseline = float(audit_targets.mean().item())
        lift = hidden_ap - frequency_baseline if math.isfinite(hidden_ap) else float("nan")
        lcb95 = _bootstrap_lcb95(
            audit_scores,
            audit_targets,
            bootstrap_samples=bootstrap_samples,
            seed=int(seed) + 1009 * label,
            confidence=confidence,
        )
        admitted = bool(math.isfinite(lcb95) and lcb95 > 0.0)
        label_lambda = (
            min(DEFAULT_PU_LAMBDA_MAX, max(0.0, lcb95 / DEFAULT_PU_LCB_SCALE))
            if admitted
            else 0.0
        )
        labels.append(
            {
                "label_id": label,
                "audit_split": "train_audit",
                "positive_count": positive_count,
                "hidden_positive_count": hidden_count,
                "hidden_positive_auprc": float(hidden_ap),
                "frequency_baseline": frequency_baseline,
                "auprc_lift": float(lift),
                "bootstrap_lcb95": float(lcb95),
                "eligible": admitted,
                "lambda": float(label_lambda),
            }
        )
        lambdas.append(float(label_lambda))

    return {
        "source_split": "train_audit",
        "audit_split": "train_audit",
        "selection": "hidden_positive_auprc_lift",
        "lambda_formula": "clip(LCB95(auprc_lift)/0.05,0,0.10)",
        "confidence": float(confidence),
        "bootstrap_samples": int(bootstrap_samples),
        "lambda": lambdas,
        "active_labels": [index for index, value in enumerate(lambdas) if value > 0.0],
        "labels": labels,
        "test_metrics_used": False,
    }


def _label_lambda(pu_lambda: Tensor, reason_dim: int, reference: Tensor) -> Tensor:
    if not isinstance(pu_lambda, Tensor) or pu_lambda.ndim != 1 or pu_lambda.numel() != reason_dim:
        raise ValueError("pu_lambda must have shape [reason_dim]")
    return pu_lambda.detach().to(device=reference.device, dtype=reference.dtype).clamp(0.0, DEFAULT_PU_LAMBDA_MAX)


def private_pu_loss(
    private_logits: Tensor,
    observed_target: Tensor,
    pu_score_value: Tensor,
    pu_lambda: Tensor,
) -> Tensor:
    """Apply detached PU targets and lambda to private reason logits only."""
    logits = _reason_matrix(private_logits, "private_logits")
    target = _reason_matrix(observed_target, "observed_target")
    score = _reason_matrix(pu_score_value, "pu_score")
    if logits.shape != target.shape or logits.shape != score.shape:
        raise ValueError("private PU tensors must share shape [batch, reason_dim]")
    target = target.detach().to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    score = score.detach().to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    lambda_by_label = _label_lambda(pu_lambda, logits.shape[1], logits).view(1, -1)
    active = lambda_by_label.gt(0.0).to(dtype=logits.dtype)
    weight = active * (target + (1.0 - target) * lambda_by_label)
    soft_target = torch.maximum(target, score)
    elements = F.binary_cross_entropy_with_logits(logits, soft_target, reduction="none")
    return (elements * weight).sum() / weight.sum().clamp_min(1.0)


def save_private_pu_loss(
    output: Mapping[str, Any],
    observed_target: Tensor,
    pu_lambda: Tensor,
    *,
    pu_score_value: Tensor | None = None,
) -> Tensor:
    """Extract only the private reason branch from a SAVE output mapping."""
    private_logits = None
    for name in (
        "reason_logits_private_direct",
        "reason_logits_private",
        "reason_logits_bench_private",
        "private_reason_logits",
    ):
        value = output.get(name)
        if isinstance(value, Tensor):
            private_logits = value
            break
    if private_logits is None:
        raise KeyError("SAVE output is missing private reason logits")
    if pu_score_value is None:
        clean = output.get("reason_logits_clean", output.get("reason_logits_global"))
        state = output.get("positive_state_probability", output.get("predicate_positive_state_probability"))
        reliability = output.get("reason_reliability", output.get("factor_reliability"))
        if not all(isinstance(value, Tensor) for value in (clean, state, reliability)):
            raise KeyError("SAVE output is missing detached PU score inputs")
        pu_score_value = pu_score(clean, state, reliability)
    return private_pu_loss(private_logits, observed_target, pu_score_value, pu_lambda)


pu_private_loss = private_pu_loss
build_save_private_pu_loss = save_private_pu_loss


def view_consistency_score(
    logits: Tensor,
    view_logits: Tensor,
    measurement: Tensor,
    view_measurement: Tensor,
    *,
    alpha: float = DEFAULT_VIEW_ALPHA,
    temperature: float = DEFAULT_VIEW_TEMPERATURE,
) -> Tensor:
    """Compute per-label cross-view consistency without changing state."""
    for name, value in (
        ("logits", logits),
        ("view_logits", view_logits),
        ("measurement", measurement),
        ("view_measurement", view_measurement),
    ):
        _reason_matrix(value, name)
    if not (logits.shape == view_logits.shape == measurement.shape == view_measurement.shape):
        raise ValueError("view consistency tensors must share shape [batch, reason_dim]")
    if float(alpha) < 0.0 or float(temperature) <= 0.0:
        raise ValueError("alpha must be non-negative and temperature must be positive")
    distance = (logits - view_logits).abs() + float(alpha) * (
        measurement - view_measurement
    ).abs()
    return torch.exp(-distance / float(temperature))


def view_consistency_loss(
    logits: Tensor,
    view_logits: Tensor,
    measurement: Tensor,
    view_measurement: Tensor,
    *,
    alpha: float = DEFAULT_VIEW_ALPHA,
    temperature: float = DEFAULT_VIEW_TEMPERATURE,
) -> Tensor:
    """Return the differentiable train-view consistency penalty."""
    return 1.0 - view_consistency_score(
        logits,
        view_logits,
        measurement,
        view_measurement,
        alpha=alpha,
        temperature=temperature,
    ).mean()


cross_view_consistency = view_consistency_score


class TrainOnlyViewConsistencyBuffer(nn.Module):
    """Train-derived EMA trust buffer; evaluation can only read it."""

    source_split = "train_only"

    def __init__(
        self,
        *,
        num_labels: int = 21,
        momentum: float = DEFAULT_VIEW_MOMENTUM,
        alpha: float = DEFAULT_VIEW_ALPHA,
        temperature: float = DEFAULT_VIEW_TEMPERATURE,
    ) -> None:
        super().__init__()
        if int(num_labels) <= 0:
            raise ValueError("num_labels must be positive")
        if not 0.0 < float(momentum) < 1.0:
            raise ValueError("view consistency momentum must be in (0, 1)")
        if float(alpha) < 0.0 or float(temperature) <= 0.0:
            raise ValueError("alpha must be non-negative and temperature must be positive")
        self.num_labels = int(num_labels)
        self.momentum = float(momentum)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.register_buffer(
            "consistency_ema", torch.ones(self.num_labels), persistent=True
        )
        self.register_buffer(
            "consistency_updates", torch.zeros((), dtype=torch.long), persistent=True
        )

    def update(
        self,
        logits: Tensor,
        view_logits: Tensor,
        measurement: Tensor,
        view_measurement: Tensor,
        *,
        split_name: str = "train_main",
        split: str | None = None,
    ) -> Tensor:
        """Update only from a training split and return detached batch values."""
        if split is not None:
            split_name = split
        _require_train_split(split_name)
        with torch.no_grad():
            score = view_consistency_score(
                logits,
                view_logits,
                measurement,
                view_measurement,
                alpha=self.alpha,
                temperature=self.temperature,
            )
            if score.shape[1] != self.num_labels:
                raise ValueError("view consistency labels do not match the buffer")
            batch_mean = score.detach().float().mean(0).to(self.consistency_ema)
            if int(self.consistency_updates.item()) == 0:
                self.consistency_ema.copy_(batch_mean)
            else:
                self.consistency_ema.mul_(self.momentum).add_(
                    batch_mean * (1.0 - self.momentum)
                )
            self.consistency_updates.add_(1)
        return score.detach()

    def read(self, *, split_name: str = "test", split: str | None = None) -> Tensor:
        """Read train-derived state without accepting test observations."""
        if split is not None:
            split_name = split
        actual = _canonical_split_name(split_name)
        if actual not in _TRAIN_SPLITS and actual not in {"test", "eval", "validation"}:
            raise ValueError(f"unknown split for view consistency read: {actual}")
        return self.consistency_ema.detach().clone()

    def forward(self, *, split_name: str = "test", update: bool = False, **kwargs: Tensor) -> Tensor:
        if update:
            return self.update(split_name=split_name, **kwargs)
        return self.read(split_name=split_name)


SAVEViewConsistencyBuffer = TrainOnlyViewConsistencyBuffer
ViewConsistencyBuffer = TrainOnlyViewConsistencyBuffer


__all__ = [
    "BBAMPrototypeBank",
    "DEFAULT_BBAM_MARGIN",
    "DEFAULT_BBAM_PROTOTYPE_MOMENTUM",
    "DEFAULT_BBAM_TAIL_COUNT",
    "DEFAULT_PU_LAMBDA_MAX",
    "DEFAULT_PU_LCB_SCALE",
    "DEFAULT_VIEW_ALPHA",
    "DEFAULT_VIEW_MOMENTUM",
    "DEFAULT_VIEW_TEMPERATURE",
    "SAVE_TAIL_REASON_ARTIFACT",
    "SAVEViewConsistencyBuffer",
    "TrainOnlyViewConsistencyBuffer",
    "ViewConsistencyBuffer",
    "admit_pu_from_train_audit",
    "balanced_angular_margin_loss",
    "balanced_bbam_loss",
    "bbam_loss",
    "build_bbam_tail_spec",
    "build_reason_reliability",
    "build_save_private_pu_loss",
    "cross_view_consistency",
    "private_pu_loss",
    "pu_private_loss",
    "pu_score",
    "save_bbam_loss",
    "save_private_pu_loss",
    "save_pu_score",
    "select_tail_reason_ids",
    "view_consistency_loss",
    "view_consistency_score",
]
