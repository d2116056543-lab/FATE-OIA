"""P12 label-wise positive-unlabeled supervision for reason-private logits."""

from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from fate_oia.losses.rael_task_losses import evidence_conditional_loss


REASON_COUNT = 21
PUBLIC_SLOT_COUNT = 20


_SAFE_ABS = 1.0e6


def _validate_reason_matrix(
    name: str,
    value: Tensor,
    *,
    batch: int | None = None,
    device: torch.device | None = None,
    validate_values: bool = False,
) -> tuple[int, torch.device]:
    if value.ndim != 2 or value.shape[1] != REASON_COUNT or value.shape[0] < 1:
        raise ValueError(f"{name} must be [B,21]")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if batch is not None and value.shape[0] != batch:
        raise ValueError(f"{name} batch mismatch")
    if device is not None and value.device != device:
        raise ValueError(f"{name} device mismatch")
    if validate_values:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        if not bool((value.float().abs() <= _SAFE_ABS).all()):
            raise ValueError(f"{name} must satisfy abs(value) <= {_SAFE_ABS:g} in debug validation")
    return value.shape[0], value.device


def _validate_probability(name: str, value: Tensor, *, validate_values: bool = False, exact_binary: bool = False) -> None:
    if not validate_values:
        return
    if not bool(((value >= 0.0) & (value <= 1.0)).all()):
        raise ValueError(f"{name} must be in [0,1]")
    if exact_binary and not bool(((value == 0.0) | (value == 1.0)).all()):
        raise ValueError(f"{name} must be exactly binary")


def reason_confidence_weights(
    unary_public_pi: Tensor,
    reliability: Tensor,
    unary_raw_contribution: Tensor,
    view_probability_one: Tensor,
    view_probability_two: Tensor,
    observability: Tensor,
    *,
    validate_values: bool = False,
) -> dict[str, Tensor]:
    """Compute the exact detached P12 c/w confidence equations in FP32."""
    if unary_public_pi.ndim != 3 or unary_public_pi.shape[1:] != (REASON_COUNT, PUBLIC_SLOT_COUNT):
        raise ValueError("unary_public_pi must be [B,21,20]")
    batch = unary_public_pi.shape[0]
    if unary_raw_contribution.shape != unary_public_pi.shape:
        raise ValueError("unary_raw_contribution must be [B,21,20]")
    if reliability.shape != (batch, PUBLIC_SLOT_COUNT):
        raise ValueError("reliability must be [B,20]")
    if not all(torch.is_floating_point(value) for value in (unary_public_pi, reliability, unary_raw_contribution)):
        raise TypeError("P12 evidence inputs must be floating point")
    if not all(value.device == unary_public_pi.device for value in (reliability, unary_raw_contribution)):
        raise ValueError("P12 evidence inputs must share a device")
    if validate_values:
        for name, value in (("unary_public_pi", unary_public_pi), ("reliability", reliability), ("unary_raw_contribution", unary_raw_contribution)):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be finite")
            if not bool((value.float().abs() <= _SAFE_ABS).all()):
                raise ValueError(f"{name} exceeds debug safe bound")
    _validate_probability("unary_public_pi", unary_public_pi, validate_values=validate_values)
    _validate_probability("reliability", reliability, validate_values=validate_values)
    matrices = {
        "view_probability_one": view_probability_one,
        "view_probability_two": view_probability_two,
        "observability": observability,
    }
    for name, value in matrices.items():
        _validate_reason_matrix(name, value, batch=batch, device=unary_public_pi.device, validate_values=validate_values)
        _validate_probability(name, value, validate_values=validate_values)

    pi = unary_public_pi.detach().float()
    rho = reliability.detach().float().unsqueeze(1)
    contribution = unary_raw_contribution.detach().float()
    c_evidence = (pi * rho * torch.sigmoid(contribution)).sum(dim=-1).clamp(0.0, 1.0)
    c_view = (1.0 - (view_probability_one.detach().float() - view_probability_two.detach().float()).abs()).clamp(0.0, 1.0)
    c_obs = observability.detach().float().clamp(0.0, 1.0)
    confidence = (c_evidence * c_view * c_obs).clamp_min(0.0).pow(1.0 / 3.0).detach()
    positive_weight = (0.4 + 0.6 * confidence).detach()
    negative_weight = (0.1 + 0.3 * c_obs * (1.0 - c_evidence)).detach()
    return {
        "c_evidence": c_evidence.detach(),
        "c_view": c_view.detach(),
        "c_obs": c_obs.detach(),
        "confidence": confidence,
        "positive_weight": positive_weight,
        "negative_weight": negative_weight,
    }


def build_pu_soft_targets(
    observed_labels: Tensor,
    evidence_probability: Tensor,
    private_probability: Tensor,
    c_view: Tensor,
    c_obs: Tensor,
    pu_lambda: Tensor,
    *,
    update_index: int,
    validate_values: bool = False,
) -> dict[str, Tensor]:
    """Build detached P12 soft labels; update zero is intentionally all-off."""
    batch, device = _validate_reason_matrix("observed_labels", observed_labels, validate_values=validate_values)
    _validate_probability("observed_labels", observed_labels, validate_values=validate_values, exact_binary=validate_values)
    for name, value in (("evidence_probability", evidence_probability), ("private_probability", private_probability), ("c_view", c_view), ("c_obs", c_obs)):
        _validate_reason_matrix(name, value, batch=batch, device=device, validate_values=validate_values)
        _validate_probability(name, value, validate_values=validate_values)
    if pu_lambda.shape != (REASON_COUNT,) or not torch.is_floating_point(pu_lambda) or pu_lambda.device != device:
        raise ValueError("pu_lambda must be floating point [21] on the label device")
    if validate_values and (not bool(torch.isfinite(pu_lambda).all()) or not bool(((pu_lambda >= 0.0) & (pu_lambda <= 0.20)).all())):
        raise ValueError("pu_lambda must be finite in [0,0.20]")
    if update_index < 0:
        raise ValueError("update_index must be nonnegative")
    effective_lambda = torch.zeros_like(pu_lambda.detach().float()) if update_index == 0 else pu_lambda.detach().float().clamp(0.0, 0.20)
    score = (
        (evidence_probability.detach().float().clamp(0.0, 1.0) * private_probability.detach().float().clamp(0.0, 1.0)).sqrt()
        * c_view.detach().float().clamp(0.0, 1.0)
        * c_obs.detach().float().clamp(0.0, 1.0)
    ).clamp(0.0, 1.0).detach()
    labels = observed_labels.detach().float()
    soft_target = (labels + (1.0 - labels) * effective_lambda.view(1, REASON_COUNT) * score).clamp(0.0, 1.0).detach()
    return {"pu_score": score, "soft_targets": soft_target, "effective_lambda": effective_lambda.detach()}


_WINDOWS_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
_WINDOWS_FORBIDDEN_COMPONENT_CHARS = frozenset('<>:"|?*')


def _canonical_windows_component(component: str, *, root_component: bool = False) -> str:
    """Normalize one Win32 path component without reading the filesystem."""
    if any(ord(character) < 32 or character in _WINDOWS_FORBIDDEN_COMPONENT_CHARS for character in component):
        raise ValueError("sample_ids path contains an invalid Win32 component")
    navigation = component.rstrip(" ")
    if navigation in {".", ".."}:
        if root_component:
            raise ValueError("sample_ids path has an invalid Win32 root component")
        return navigation
    normalized = component.rstrip(". ")
    if not normalized:
        if root_component:
            raise ValueError("sample_ids path has an invalid Win32 root component")
        return normalized
    return normalized


def _resolve_windows_components(components: Sequence[str], *, root_name: str) -> list[str]:
    """Resolve dots lexically and forbid escaping the declared root."""
    resolved: list[str] = []
    for raw_component in components:
        if raw_component == "":
            continue
        component = _canonical_windows_component(raw_component)
        if component == ".":
            continue
        if component == "..":
            if not resolved:
                raise ValueError(f"sample_ids path cannot traverse above {root_name}")
            resolved.pop()
            continue
        if not component:
            raise ValueError("sample_ids path has an empty Win32 component")
        resolved.append(component)
    return resolved


def _canonical_windows_image_path(sample_id: str) -> str:
    """Pure lexical Windows path canonicalizer with explicit drive/UNC/relative roots."""
    if not sample_id:
        raise ValueError("sample_ids strings must be nonempty")
    if not sample_id.isascii():
        raise ValueError("sample_ids paths must be ASCII Windows dataset-image paths")
    if sample_id != sample_id.strip():
        raise ValueError("sample_ids strings must not have leading or trailing whitespace")
    normalized = sample_id.replace("\\", "/").lower()

    if normalized.startswith("//"):
        unc_components = [part for part in normalized[2:].split("/") if part]
        if len(unc_components) < 3:
            raise ValueError("sample_ids UNC paths require server, share, and image name")
        server = _canonical_windows_component(unc_components[0], root_component=True)
        share = _canonical_windows_component(unc_components[1], root_component=True)
        components = _resolve_windows_components(unc_components[2:], root_name="the UNC share root")
        namespace = f"path:unc:{server}/{share}"
    elif len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0]
        if not drive.isascii() or not drive.isalpha() or len(normalized) < 3 or normalized[2] != "/":
            raise ValueError("sample_ids drive paths must be absolute Win32 paths such as C:/image.jpg")
        components = _resolve_windows_components(normalized[3:].split("/"), root_name="the drive root")
        namespace = f"path:drive:{drive}:"
    elif normalized.startswith("/"):
        raise ValueError("sample_ids rooted paths must include a Win32 drive or UNC share")
    else:
        components = _resolve_windows_components(normalized.split("/"), root_name="the relative root")
        namespace = "path:relative"

    if not components:
        raise ValueError("sample_ids paths must name an image file")
    if not any(components[-1].endswith(suffix) for suffix in _WINDOWS_IMAGE_SUFFIXES):
        raise ValueError("sample_ids strings must be Windows image paths with a supported image suffix")
    return f"{namespace}/{'/'.join(components)}"


def canonicalize_sample_id(sample_id: str | int) -> str:
    """Canonical, type-preserving hash identity for integer or Windows image-path IDs."""
    if isinstance(sample_id, bool):
        raise TypeError("sample_ids must not contain bool")
    if isinstance(sample_id, numbers.Integral):
        return f"int:{int(sample_id)}"
    if isinstance(sample_id, str):
        return _canonical_windows_image_path(sample_id)
    raise TypeError("sample_ids entries must be non-bool integers or Windows image-path strings")


def canonical_sample_id_hash(canonical_id: str, *, label_index: int, seed: int) -> bytes:
    """Versioned SHA256 key for deterministic per-label audit ordering."""
    return hashlib.sha256(f"rael-p12-id-v1\x00{seed}\x00{label_index}\x00{canonical_id}".encode("utf-8")).digest()


def _canonical_sample_ids(sample_ids: Sequence[str | int] | Tensor, *, batch: int) -> list[str]:
    """Normalize strict sample identities; row index is never an identity."""
    if isinstance(sample_ids, Tensor):
        if sample_ids.ndim != 1:
            raise ValueError("sample_ids tensor must be [B]")
        if sample_ids.dtype == torch.bool or torch.is_floating_point(sample_ids):
            raise TypeError("sample_ids tensor must have a non-bool integer dtype")
        values: list[Any] = sample_ids.detach().cpu().tolist()
    else:
        if isinstance(sample_ids, (str, bytes)):
            raise TypeError("sample_ids must be a sequence, not a scalar string")
        values = list(sample_ids)
    if len(values) != batch:
        raise ValueError("sample_ids length must equal batch size")
    normalized = [canonicalize_sample_id(value) for value in values]
    if len(set(normalized)) != batch:
        raise ValueError("sample_ids must be unique after canonicalization")
    return normalized


def round_half_up(value: float) -> int:
    """Positive-domain decimal rounding used by the fixed 30% audit protocol."""
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("round_half_up expects a finite nonnegative value")
    return int(math.floor(value + 0.5))


def deterministic_known_positive_mask(
    observed_labels: Tensor,
    *,
    sample_ids: Sequence[str | int] | Tensor,
    seed: int,
    fraction: float = 0.30,
    label_indices: Sequence[int] | None = None,
) -> Tensor:
    """Hide exactly round-half-up(fraction*n) positives per label by stable ID."""
    batch, device = _validate_reason_matrix("observed_labels", observed_labels, validate_values=True)
    _validate_probability("observed_labels", observed_labels, validate_values=True, exact_binary=True)
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0,1)")
    ids = _canonical_sample_ids(sample_ids, batch=batch)
    labels = observed_labels.detach().float()
    hidden = torch.zeros((batch, REASON_COUNT), device=device, dtype=torch.bool)
    selected_labels = list(range(REASON_COUNT)) if label_indices is None else list(label_indices)
    if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 or index >= REASON_COUNT for index in selected_labels):
        raise ValueError("label_indices must contain unique integer reason indices")
    if len(set(selected_labels)) != len(selected_labels):
        raise ValueError("label_indices must be unique")
    for label_index in selected_labels:
        positive_rows = torch.nonzero(labels[:, label_index] > 0.5, as_tuple=False).flatten().detach().cpu().tolist()
        hide_count = round_half_up(fraction * len(positive_rows))
        ranked = sorted(
            positive_rows,
            key=lambda row: (
                canonical_sample_id_hash(ids[row], label_index=label_index, seed=seed),
                ids[row],
            ),
        )
        if hide_count:
            chosen = torch.tensor(ranked[:hide_count], dtype=torch.long, device=device)
            hidden[chosen, label_index] = True
    return hidden.detach()


def average_precision_binary(scores: Tensor, targets: Tensor) -> Tensor:
    """Tie-grouped binary AUPRC, invariant to row order inside equal scores."""
    if scores.ndim != 1 or targets.ndim != 1 or scores.shape != targets.shape or scores.numel() < 1:
        raise ValueError("scores and targets must be same nonempty [N]")
    if not torch.is_floating_point(scores) or not torch.is_floating_point(targets):
        raise TypeError("scores and targets must be floating point")
    if scores.device.type != "cpu" or targets.device.type != "cpu":
        raise ValueError("AUPRC is a CPU diagnostic and requires CPU inputs")
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("scores and targets must be finite")
    _validate_probability("targets", targets, validate_values=True, exact_binary=True)
    score = scores.detach().float()
    target = targets.detach().float()
    positives = target.sum()
    if positives <= 0:
        return torch.tensor(0.0)
    order = torch.argsort(score, descending=True, stable=True)
    sorted_score = score[order]
    sorted_target = target[order]
    group_start = torch.ones(sorted_score.numel(), dtype=torch.bool)
    group_start[1:] = sorted_score[1:] != sorted_score[:-1]
    starts = torch.nonzero(group_start, as_tuple=False).flatten()
    ends = torch.cat((starts[1:] - 1, torch.tensor([sorted_score.numel() - 1], dtype=torch.long)))
    cumulative_positive = sorted_target.cumsum(0)
    cumulative_count = ends.to(torch.float32) + 1.0
    previous_positive = torch.cat((torch.zeros(1), cumulative_positive[ends[:-1]]))
    group_positive = cumulative_positive[ends] - previous_positive
    precision_at_group_end = cumulative_positive[ends] / cumulative_count
    return (precision_at_group_end * group_positive).sum() / positives


def paired_bootstrap_lcb95(
    pu_scores: Tensor,
    baseline_scores: Tensor,
    targets: Tensor,
    *,
    seed: int,
    resample_count: int,
) -> dict[str, Any]:
    """One-sided paired-bootstrap 5th percentile for a single label's delta AP."""
    if resample_count < 1:
        raise ValueError("resample_count must be positive")
    if pu_scores.ndim != 1 or baseline_scores.ndim != 1 or targets.ndim != 1:
        raise ValueError("paired bootstrap inputs must be [N]")
    if pu_scores.shape != baseline_scores.shape or pu_scores.shape != targets.shape or targets.numel() < 1:
        raise ValueError("paired bootstrap inputs must have the same nonempty shape")
    if not all(torch.is_floating_point(value) for value in (pu_scores, baseline_scores, targets)):
        raise TypeError("paired bootstrap inputs must be floating point")
    if any(value.device.type != "cpu" for value in (pu_scores, baseline_scores, targets)):
        raise ValueError("paired bootstrap is a CPU diagnostic and requires CPU inputs")
    if not all(bool(torch.isfinite(value).all()) for value in (pu_scores, baseline_scores, targets)):
        raise ValueError("paired bootstrap inputs must be finite")
    _validate_probability("targets", targets, validate_values=True, exact_binary=True)
    pu = pu_scores.detach().float()
    baseline = baseline_scores.detach().float()
    target = targets.detach().float()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    digest = hashlib.sha256()
    deltas: list[float] = []
    skipped = 0
    for _ in range(resample_count):
        indices = torch.randint(target.numel(), (target.numel(),), generator=generator)
        digest.update(indices.numpy().tobytes())
        sampled_target = target[indices]
        if sampled_target.sum() <= 0.0:
            skipped += 1
            continue
        # The single sampled index vector is deliberately used for both arms.
        delta = average_precision_binary(pu[indices], sampled_target) - average_precision_binary(baseline[indices], sampled_target)
        deltas.append(float(delta))
    valid = len(deltas)
    lcb = torch.tensor(float("nan"), dtype=torch.float32) if valid == 0 else torch.quantile(torch.tensor(deltas, dtype=torch.float32), 0.05)
    return {
        "lcb95": lcb,
        "percentile": 0.05,
        "valid_resample_count": valid,
        "skipped_resample_count": skipped,
        "resample_count": resample_count,
        "resample_index_digest": digest.hexdigest(),
    }


@torch.no_grad()
def labelwise_pu_audit(
    observed_labels: Tensor,
    pu_scores: Tensor,
    visual_baseline_scores: Tensor,
    *,
    sample_ids: Sequence[str | int] | Tensor,
    split: str,
    update_index: int,
    seed: int = 20260725,
    hide_fraction: float = 0.30,
    min_positive_count: int = 20,
    resample_count: int = 200,
) -> dict[str, Any]:
    """Train-audit-only label gate with exact ID hiding and paired one-sided LCB."""
    if split != "train_audit":
        raise ValueError("P12 PU audit is restricted to split=train_audit")
    batch, device = _validate_reason_matrix("observed_labels", observed_labels, validate_values=True)
    if device.type != "cpu":
        raise ValueError("labelwise PU audit is a CPU diagnostic and requires CPU inputs")
    _validate_probability("observed_labels", observed_labels, validate_values=True, exact_binary=True)
    for name, value in (("pu_scores", pu_scores), ("visual_baseline_scores", visual_baseline_scores)):
        _validate_reason_matrix(name, value, batch=batch, device=device, validate_values=True)
    if update_index < 0 or min_positive_count < 1 or resample_count < 1:
        raise ValueError("invalid P12 audit configuration")
    labels = observed_labels.detach().float()
    _canonical_sample_ids(sample_ids, batch=batch)
    positive_count = labels.sum(dim=0).to(dtype=torch.int64)
    empty_float = torch.zeros(REASON_COUNT, device=device, dtype=torch.float32)
    empty_int = torch.zeros(REASON_COUNT, device=device, dtype=torch.int64)

    def all_off(reason: str, *, hidden: Tensor | None = None, lcb_value: float = float("nan")) -> dict[str, Any]:
        return {
            "hidden_positive_mask": torch.zeros((batch, REASON_COUNT), device=device, dtype=torch.bool) if hidden is None else hidden.detach(),
            "positive_count": positive_count.detach(),
            "hidden_positive_count": empty_int.detach(),
            "pu_auprc": empty_float.detach(),
            "baseline_auprc": empty_float.detach(),
            "delta_auprc": empty_float.detach(),
            "lcb95_delta_auprc": torch.full((REASON_COUNT,), lcb_value, device=device, dtype=torch.float32),
            "pu_lambda": empty_float.detach(),
            "active": torch.zeros(REASON_COUNT, device=device, dtype=torch.bool),
            "active_reason": [reason] * REASON_COUNT,
            "bootstrap_valid_resample_count": empty_int.detach(),
            "bootstrap_skipped_resample_count": empty_int.detach(),
            "bootstrap_resample_index_digest": ["not_run"] * REASON_COUNT,
            "statistics": "one-sided paired bootstrap 5th-percentile lower bound over fixed train-audit rows",
        }

    # Update zero is an intentional all-off state, not an expensive diagnostic.
    if update_index == 0:
        return all_off("epoch0_all_off")
    candidate_labels = [index for index in range(REASON_COUNT) if int(positive_count[index]) >= min_positive_count]
    # A common sparse-tail audit case: no label is eligible, so skip hashing,
    # AUPRC, and bootstrap entirely.
    if not candidate_labels:
        return all_off("count_below_min")
    hidden = deterministic_known_positive_mask(labels, sample_ids=sample_ids, seed=seed, fraction=hide_fraction, label_indices=candidate_labels)
    delta = torch.zeros(REASON_COUNT, device=device, dtype=torch.float32)
    lcb = torch.full((REASON_COUNT,), float("nan"), device=device, dtype=torch.float32)
    pu_ap = torch.zeros(REASON_COUNT, device=device, dtype=torch.float32)
    baseline_ap = torch.zeros(REASON_COUNT, device=device, dtype=torch.float32)
    active = torch.zeros(REASON_COUNT, device=device, dtype=torch.bool)
    lambda_values = torch.zeros(REASON_COUNT, device=device, dtype=torch.float32)
    valid_resamples = torch.zeros(REASON_COUNT, device=device, dtype=torch.int64)
    skipped_resamples = torch.zeros(REASON_COUNT, device=device, dtype=torch.int64)
    index_digests = ["not_run"] * REASON_COUNT
    reasons = ["count_below_min"] * REASON_COUNT
    for label_index in candidate_labels:
        hidden_label = hidden[:, label_index]
        evaluation = hidden_label | (labels[:, label_index] <= 0.5)
        target = hidden_label[evaluation].float().cpu()
        pu = pu_scores.detach().float()[evaluation, label_index].cpu()
        baseline = visual_baseline_scores.detach().float()[evaluation, label_index].cpu()
        pu_ap[label_index] = average_precision_binary(pu, target).to(device)
        baseline_ap[label_index] = average_precision_binary(baseline, target).to(device)
        delta[label_index] = pu_ap[label_index] - baseline_ap[label_index]
        bootstrap = paired_bootstrap_lcb95(pu, baseline, target, seed=seed + 104_729 * label_index, resample_count=resample_count)
        lcb[label_index] = bootstrap["lcb95"].to(device)
        valid_resamples[label_index] = bootstrap["valid_resample_count"]
        skipped_resamples[label_index] = bootstrap["skipped_resample_count"]
        index_digests[label_index] = bootstrap["resample_index_digest"]
        if int(valid_resamples[label_index]) == 0:
            reasons[label_index] = "no_valid_bootstrap"
        elif lcb[label_index] <= 0.0:
            reasons[label_index] = "lcb_not_positive"
        else:
            active[label_index] = True
            lambda_values[label_index] = (delta[label_index] / 0.05).clamp(0.0, 0.20)
            reasons[label_index] = "active"
    return {
        "hidden_positive_mask": hidden.detach(),
        "positive_count": positive_count.detach(),
        "hidden_positive_count": hidden.sum(dim=0).to(dtype=torch.int64).detach(),
        "pu_auprc": pu_ap.detach(),
        "baseline_auprc": baseline_ap.detach(),
        "delta_auprc": delta.detach(),
        "lcb95_delta_auprc": lcb.detach(),
        "pu_lambda": lambda_values.detach(),
        "active": active.detach(),
        "active_reason": reasons,
        "bootstrap_valid_resample_count": valid_resamples.detach(),
        "bootstrap_skipped_resample_count": skipped_resamples.detach(),
        "bootstrap_resample_index_digest": index_digests,
        "statistics": "one-sided paired bootstrap 5th-percentile lower bound over fixed train-audit rows",
    }


def reason_private_pu_loss(
    private_logits: Tensor,
    soft_targets: Tensor,
    positive_weight: Tensor,
    negative_weight: Tensor,
    *,
    validate_values: bool = False,
) -> Tensor:
    """The only P12 differentiable path: reason-private logits receive detached supervision."""
    _validate_reason_matrix("private_logits", private_logits, validate_values=validate_values)
    for name, value in (("soft_targets", soft_targets), ("positive_weight", positive_weight), ("negative_weight", negative_weight)):
        _validate_reason_matrix(name, value, batch=private_logits.shape[0], device=private_logits.device, validate_values=validate_values)
    return evidence_conditional_loss(
        private_logits,
        soft_targets.detach(),
        positive_weight.detach(),
        negative_weight.detach(),
        validate_values=validate_values,
    )


__all__ = [
    "PUBLIC_SLOT_COUNT",
    "REASON_COUNT",
    "average_precision_binary",
    "build_pu_soft_targets",
    "deterministic_known_positive_mask",
    "labelwise_pu_audit",
    "paired_bootstrap_lcb95",
    "reason_confidence_weights",
    "reason_private_pu_loss",
    "round_half_up",
]
