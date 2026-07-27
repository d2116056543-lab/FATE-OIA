"""Strict test-only RAEL evaluator with one field encode and P18-ready artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Any

import torch
import yaml
from torch import Tensor, nn

from fate_oia.losses.rael_pu_losses import canonicalize_sample_id
from fate_oia.engine.export_rael_cases import RAELCaseExportCollector
from fate_oia.models.rael_oia_model import BRANCH_NAMES
from fate_oia.utils.rael_posthoc_calibration import (
    apply_posthoc_calibration,
    serialize_calibration_result,
)
from fate_oia.utils.rael_schema import load_reason_semantic_schema


_ACTION_DIM = 4
_REASON_DIM = 21
_PLACEHOLDER_LABEL_RE = re.compile(r"^(?:action|reason)[_ -]?\d+$", re.IGNORECASE)
_REJECTED_LABEL_NAMES = frozenset({"unknown", "tbd", "placeholder"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_CASE_EXPORT_CONSUME_ARGUMENTS = frozenset(
    {
        "file_names",
        "action_targets",
        "reason_targets",
        "outputs",
        "action_calibration",
        "reason_calibration",
        "action_names",
        "reason_names",
    }
)


def _epoch_diagnostic_row(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Seal scalar/tiny-matrix diagnostics from this exact test decode.

    This does not run another decoder.  P21's P18 adapter aggregates only
    these detached values, so its epoch records stay attributable to the
    same test tensors used for raw/deploy metrics.
    """

    required = {
        "slot_masks", "slot_area", "slot_reliability", "layer_weights_action",
        "layer_weights_reason", "layer_weights_slots", "action_unary_contributions",
        "reason_unary_contributions", "action_pairwise_contributions",
        "reason_pairwise_contributions", "action_global_contribution",
        "reason_global_contribution", "named_contribution_ratio",
        "latent_contribution_ratio", "positive_contribution", "negative_contribution",
        "null_mass", "diagnostics",
    }
    missing = required.difference(outputs)
    if missing:
        raise ValueError(f"P18 epoch diagnostics missing {sorted(missing)}")
    masks = outputs["slot_masks"]
    area = outputs["slot_area"]
    reliability = outputs["slot_reliability"]
    if not all(isinstance(value, Tensor) for value in (masks, area, reliability)):
        raise TypeError("P18 epoch slot diagnostics must be tensors")
    if masks.ndim != 4 or masks.shape[1] != 20 or area.shape[:2] != masks.shape[:2] or reliability.shape[:2] != masks.shape[:2]:
        raise ValueError("P18 epoch slot diagnostic shapes are invalid")

    def mean_tensor(value: Any, *, expected: tuple[int, ...] | None = None) -> Any:
        if not isinstance(value, Tensor):
            raise TypeError("P18 diagnostic value must be tensor")
        result = value.detach().float().mean(dim=0)
        if expected is not None and tuple(result.shape) != expected:
            raise ValueError(f"P18 diagnostic matrix has {tuple(result.shape)}, expected {expected}")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("P18 diagnostic tensor is non-finite")
        return result.cpu().tolist()

    def rms(value: Any) -> float:
        if not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError("P18 contribution diagnostic must be finite tensor")
        return float(value.detach().float().square().mean().sqrt().item())

    def family_mapping(name: str) -> dict[str, float]:
        value = outputs[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"P18 diagnostic {name} must be a mapping")
        converted: dict[str, float] = {}
        for family in ("action", "reason"):
            item = value.get(family)
            if not isinstance(item, Tensor) or not bool(torch.isfinite(item).all()):
                raise ValueError(f"P18 diagnostic {name}.{family} must be finite tensor")
            converted[family] = float(item.detach().float().mean().item())
        return converted

    def family_vector(name: str) -> dict[str, list[float]]:
        value = outputs[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"P18 diagnostic {name} must be a mapping")
        widths = {"action": 4, "reason": 21}
        converted: dict[str, list[float]] = {}
        for family, width in widths.items():
            item = value.get(family)
            if not isinstance(item, Tensor) or item.ndim != 2 or item.shape[1] != width:
                raise ValueError(f"P18 diagnostic {name}.{family} must be [B,{width}]")
            mean = item.detach().float().mean(dim=0)
            if not bool(torch.isfinite(mean).all()):
                raise ValueError(f"P18 diagnostic {name}.{family} is non-finite")
            converted[family] = mean.cpu().tolist()
        return converted

    diagnostics = outputs["diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise TypeError("P18 model diagnostics must be a mapping")
    reconstruction = max(
        float(diagnostics["action_reconstruction_max_error"].detach().float().item()),
        float(diagnostics["reason_reconstruction_max_error"].detach().float().item()),
    )
    flattened_masks = masks.detach().float().flatten(start_dim=-2).clamp_min(0.0)
    normalized_masks = flattened_masks / flattened_masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    slot_entropy = float((-(normalized_masks * normalized_masks.clamp_min(1.0e-8).log()).sum(dim=-1)).mean().item())
    ledger_masks = outputs.get("ledger_slot_masks_diagnostic")
    type_probs = outputs.get("slot_type_probs")
    if not isinstance(ledger_masks, Tensor) or ledger_masks.shape != masks.shape:
        raise ValueError("P18 epoch diagnostics require canonical/ledger slot masks with matching shape")
    if not isinstance(type_probs, Tensor) or type_probs.ndim != 3 or type_probs.shape[:2] != (masks.shape[0], 12):
        raise ValueError("P18 epoch diagnostics require entity slot type probabilities")
    intersection = (masks[:, :12].detach().float() * ledger_masks[:, :12].detach().float()).sum(dim=(-2, -1))
    union = (masks[:, :12].detach().float() + ledger_masks[:, :12].detach().float() - masks[:, :12].detach().float() * ledger_masks[:, :12].detach().float()).sum(dim=(-2, -1)).clamp_min(1.0e-8)
    slot_iou = float((intersection / union).mean().item())
    type_entropy = float((-(type_probs.detach().float().clamp_min(1.0e-8) * type_probs.detach().float().clamp_min(1.0e-8).log()).sum(dim=-1)).mean().item())
    all_layer_weights = torch.cat(
        (outputs["layer_weights_action"].detach().float().reshape(-1, 4), outputs["layer_weights_reason"].detach().float().reshape(-1, 4), outputs["layer_weights_slots"].detach().float().reshape(-1, 4)),
        dim=0,
    ).clamp_min(1.0e-8)
    layer_entropy = float((-(all_layer_weights * all_layer_weights.log()).sum(dim=-1)).mean().item())
    collapse = diagnostics.get("collapse")
    collapsed = collapse.get("layer_collapse_fail") if isinstance(collapse, Mapping) else None
    if not isinstance(collapsed, Tensor) or collapsed.dtype != torch.bool:
        raise ValueError("P18 epoch diagnostics require formal collapse state")
    return {
        "sample_count": int(masks.shape[0]),
        "slot_mass": {
            "named": float(masks[:, :12].detach().float().mean().item()),
            "latent": float(masks[:, 17:20].detach().float().mean().item()),
            "background": float(outputs["background_mask"].detach().float().mean().item()),
        },
        "slot_area_mean": float(area.detach().float().mean().item()),
        "slot_area_std": float(area.detach().float().std(unbiased=False).item()),
        "slot_reliability_mean": float(reliability.detach().float().mean().item()),
        "slot_entropy": slot_entropy,
        "slot_iou": slot_iou,
        "entity_type_entropy": type_entropy,
        "layer_entropy": layer_entropy,
        "collapse": bool(collapsed.any().item()),
        "action_layer_weights": mean_tensor(outputs["layer_weights_action"], expected=(4, 4)),
        "reason_layer_weights": mean_tensor(outputs["layer_weights_reason"], expected=(21, 4)),
        "slot_layer_weights": mean_tensor(outputs["layer_weights_slots"][:, :20], expected=(20, 4)),
        "unary_rms": {"action": rms(outputs["action_unary_contributions"]), "reason": rms(outputs["reason_unary_contributions"])},
        "pairwise_rms": {"action": rms(outputs["action_pairwise_contributions"]), "reason": rms(outputs["reason_pairwise_contributions"])},
        "global_rms": {"action": rms(outputs["action_global_contribution"]), "reason": rms(outputs["reason_global_contribution"])},
        "named_ratio": family_mapping("named_contribution_ratio"),
        "named_ratio_by_target": family_vector("named_contribution_ratio"),
        "latent_ratio": family_mapping("latent_contribution_ratio"),
        "latent_ratio_by_target": family_vector("latent_contribution_ratio"),
        "positive": family_mapping("positive_contribution"),
        "negative": family_mapping("negative_contribution"),
        "null_mass": family_mapping("null_mass"),
        "reconstruction_error": reconstruction,
        "active_pair_count": int(outputs["action_pair_indices"].shape[0]),
        "total_pair_count": int(outputs["reason_pair_indices"].shape[0]),
        "pu_scores": mean_tensor(outputs["pu_scores"], expected=(21,)),
        "pu_active_labels": outputs["pu_active_labels"].detach().cpu().to(dtype=torch.bool).tolist(),
    }


def binary_roc_auc(scores: Tensor, targets: Tensor) -> float:
    """Tie-correct AUC primitive; undefined one-class inputs deliberately return NaN."""

    scores = scores.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    if scores.shape != targets.shape or scores.numel() < 1:
        raise ValueError("AUC scores and targets must be matching nonempty vectors")
    if not bool(torch.isfinite(scores).all()) or not bool(
        ((targets == 0.0) | (targets == 1.0)).all()
    ):
        raise ValueError("AUC inputs must be finite scores and binary targets")
    positives = int(targets.sum().item())
    negatives = targets.numel() - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    ranks = torch.empty_like(sorted_scores, dtype=torch.float64)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (float(start + 1) + float(end)) / 2.0
        start = end
    original_ranks = torch.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = original_ranks[targets.bool()].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0).item()
        / float(positives * negatives)
    )


def binary_average_precision_tie_stable(scores: Tensor, targets: Tensor) -> float:
    """Compute AP by whole score groups so input row order cannot break ties."""

    scores = scores.detach().float().flatten().cpu()
    targets = targets.detach().float().flatten().cpu()
    if scores.shape != targets.shape or scores.numel() < 1:
        raise ValueError("AP scores and targets must be matching nonempty vectors")
    if not bool(torch.isfinite(scores).all()) or not bool(
        ((targets == 0.0) | (targets == 1.0)).all()
    ):
        raise ValueError("AP inputs must be finite scores and binary targets")
    positive_total = int(targets.sum().item())
    if positive_total == 0:
        return float("nan")

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    cumulative_positive = 0.0
    consumed = 0
    precision_sum = 0.0
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_positive = float(sorted_targets[start:end].sum().item())
        consumed += end - start
        cumulative_positive += group_positive
        if group_positive:
            precision_sum += (cumulative_positive / float(consumed)) * group_positive
        start = end
    return precision_sum / float(positive_total)


def _require_label_names(
    names: Sequence[str], *, expected: int, context: str
) -> tuple[str, ...]:
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
        raise TypeError(f"{context} must be a {expected}-name schema sequence")
    if len(names) != expected:
        raise ValueError(f"{context} must contain exactly {expected} names")
    normalized: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{context} names must be nonempty strings")
        clean = name.strip()
        key = clean.casefold()
        if _PLACEHOLDER_LABEL_RE.fullmatch(clean) or key in _REJECTED_LABEL_NAMES:
            raise ValueError(f"{context} rejects placeholder label name {clean!r}")
        normalized.append(clean)
    if len({name.casefold() for name in normalized}) != expected:
        raise ValueError(f"{context} names must be unique")
    return tuple(normalized)


def _load_action_semantic_names(path: str | Path) -> tuple[str, ...]:
    schema_path = Path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"RAEL action schema is missing: {schema_path}")
    payload = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("RAEL action schema must be a mapping")
    if (
        payload.get("schema_version") != 1
        or payload.get("role_application") != "soft_prior_only"
        or payload.get("hard_action_masks") is not False
    ):
        raise ValueError("RAEL action schema semantic contract is invalid")
    action_order = payload.get("action_order")
    actions = payload.get("actions")
    if not isinstance(action_order, list) or not isinstance(actions, list) or len(actions) != _ACTION_DIM:
        raise ValueError("RAEL action schema must contain exactly four ordered actions")
    if not all(isinstance(row, Mapping) for row in actions):
        raise ValueError("RAEL action schema actions must be mappings")
    ids = [row.get("id") for row in actions]
    if ids != list(range(_ACTION_DIM)):
        raise ValueError("RAEL action schema IDs must be exactly ordered 0..3")
    names = _require_label_names(
        [row.get("name") for row in actions],
        expected=_ACTION_DIM,
        context="RAEL action schema",
    )
    if action_order != list(names):
        raise ValueError("RAEL action schema action_order must exactly match ordered IDs")
    return names


def _load_reason_semantic_names(path: str | Path) -> tuple[str, ...]:
    rows = load_reason_semantic_schema(path)
    if len(rows) != _REASON_DIM or tuple(row.id for row in rows) != tuple(range(_REASON_DIM)):
        raise ValueError("RAEL reason schema IDs must be exactly ordered 0..20")
    return _require_label_names(
        [row.name for row in rows], expected=_REASON_DIM, context="RAEL reason schema"
    )


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return value


def _require_calibration_source_hash(
    calibration: Mapping[str, Any], *, expected: str, context: str
) -> None:
    source = calibration.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{context} source descriptor is missing")
    observed = _require_sha256(source.get("split_hash"), context=f"{context} source split hash")
    if observed != expected:
        raise ValueError(f"{context} source split hash does not match expected train_calib split")


class _P12CompactSplitHasher:
    """Stream the exact P15 compact JSON array hash without retaining image batches."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._first = True
        self._closed = False
        self._canonical_ids: set[str] = set()

    def add(self, sample_id: str) -> None:
        if self._closed:
            raise RuntimeError("P12 split hasher is closed")
        canonical_id = canonicalize_sample_id(sample_id)
        if canonical_id in self._canonical_ids:
            raise ValueError("test file_names must be unique after P12 canonicalization")
        self._canonical_ids.add(canonical_id)
        if not self._first:
            self._digest.update(b",")
        self._digest.update(
            json.dumps(canonical_id, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        )
        self._first = False

    def hexdigest(self) -> str:
        if not self._closed:
            self._digest.update(b"]")
            self._closed = True
        return self._digest.hexdigest()


def _require_finite_binary_targets(targets: Tensor, *, expected: int, context: str) -> None:
    if targets.ndim != 2 or targets.shape[1] != expected:
        raise ValueError(f"{context} must be [B,{expected}]")
    if not bool(torch.isfinite(targets).all()) or not bool(
        ((targets == 0.0) | (targets == 1.0)).all()
    ):
        raise ValueError(f"{context} must be finite binary targets")


def _seal_tensor(value: Tensor, *, dtype: torch.dtype, shape: tuple[int, int], context: str) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != shape:
        raise ValueError(f"{context} must have shape {shape}")
    sealed = value.detach().to(device="cpu", dtype=dtype).contiguous()
    if sealed.requires_grad or not bool(torch.isfinite(sealed).all()):
        raise ValueError(f"{context} must be detached finite tensor data")
    return sealed


def _require_calibration_descriptor(value: Mapping[str, Any], *, targets: int, context: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a P15 calibration mapping")
    if value.get("fit_split") != "train_calib" or value.get("targets") != targets:
        raise ValueError(f"{context} must be a {targets}-target train_calib P15 calibration")


def _diagnostic_dino_call_count(outputs: Mapping[str, Any]) -> None:
    diagnostics = outputs.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or "dino_call_count" not in diagnostics:
        raise ValueError("RAEL decode diagnostics.dino_call_count is required")
    value = diagnostics["dino_call_count"]
    if isinstance(value, Tensor):
        if value.numel() != 1 or value.is_complex() or not bool(torch.isfinite(value.detach()).all()):
            raise ValueError("diagnostics.dino_call_count must be a finite scalar")
        resolved = float(value.detach().cpu().item())
    elif isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("diagnostics.dino_call_count must be an int or scalar tensor")
    else:
        resolved = float(value)
    if resolved != 1.0:
        raise ValueError("diagnostics.dino_call_count must equal exactly one")


def _require_case_export_collector(
    case_collector: object | None,
    case_export_provenance: Mapping[str, Any] | None,
) -> object | None:
    """Accept the P19 collector or an exact structural equivalent before DINO work."""

    if (case_collector is None) != (case_export_provenance is None):
        raise ValueError(
            "case_collector and case_export_provenance must be provided together or omitted together"
        )
    if case_collector is None:
        return None
    if isinstance(case_collector, RAELCaseExportCollector):
        return case_collector

    consume = getattr(case_collector, "consume", None)
    finalize = getattr(case_collector, "finalize", None)
    if not callable(consume) or not callable(finalize):
        raise TypeError(
            "case_collector must be RAELCaseExportCollector or implement strict consume/finalize protocol"
        )
    try:
        consume_parameters = tuple(inspect.signature(consume).parameters.values())
        finalize_parameters = tuple(inspect.signature(finalize).parameters.values())
    except (TypeError, ValueError) as error:
        raise TypeError(
            "case_collector must expose inspectable strict consume/finalize protocol"
        ) from error
    if (
        len(consume_parameters) != len(_CASE_EXPORT_CONSUME_ARGUMENTS)
        or {parameter.name for parameter in consume_parameters} != _CASE_EXPORT_CONSUME_ARGUMENTS
        or any(parameter.kind is not inspect.Parameter.KEYWORD_ONLY for parameter in consume_parameters)
        or any(parameter.default is not inspect.Parameter.empty for parameter in consume_parameters)
        or len(finalize_parameters) != 1
        or finalize_parameters[0].name != "provenance"
        or finalize_parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        or finalize_parameters[0].default is not inspect.Parameter.empty
    ):
        raise TypeError(
            "case_collector must be RAELCaseExportCollector or implement strict consume/finalize protocol"
        )
    return case_collector


def _require_batch(batch: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor, tuple[str, ...]]:
    if not isinstance(batch, Mapping) or batch.get("split") != "test":
        raise ValueError("RAEL evaluator accepts only split='test'")
    images = batch.get("images")
    action = batch.get("action_targets")
    reason = batch.get("reason_targets")
    names = batch.get("file_names")
    if not all(isinstance(value, Tensor) for value in (images, action, reason)):
        raise TypeError("test batch must contain image/action/reason tensors")
    if images.ndim < 1 or images.shape[0] < 1 or not bool(torch.isfinite(images).all()):
        raise ValueError("test images must be finite and nonempty")
    _require_finite_binary_targets(action, expected=_ACTION_DIM, context="action_targets")
    _require_finite_binary_targets(reason, expected=_REASON_DIM, context="reason_targets")
    if images.shape[0] != action.shape[0] or reason.shape[0] != action.shape[0]:
        raise ValueError("test batch sizes must match")
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or len(names) != action.shape[0]:
        raise ValueError("test batch must include one file name per row")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("test file names must be nonempty strings")
    return images, action, reason, tuple(names)


def _family_metrics_and_rows(
    *,
    logits: Tensor,
    targets: Tensor,
    decisions: Tensor,
    thresholds: Tensor,
    label_names: Sequence[str],
    decision_source: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if logits.ndim != 2 or logits.shape != targets.shape or logits.shape[0] < 1:
        raise ValueError("metric logits and targets must be matching [B,K]")
    if decisions.shape != logits.shape or thresholds.shape != (logits.shape[1],):
        raise ValueError("metric decisions and thresholds must match logits")
    if len(label_names) != logits.shape[1]:
        raise ValueError("metric label schema must match logit width")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("metric inputs must be finite")
    prediction = decisions.detach().to(dtype=torch.float32, device="cpu")
    targets = targets.detach().to(dtype=torch.float32, device="cpu")
    logits = logits.detach().to(dtype=torch.float32, device="cpu")
    tp = (prediction * targets).sum(0)
    fp = (prediction * (1.0 - targets)).sum(0)
    fn = ((1.0 - prediction) * targets).sum(0)
    denominator = 2.0 * tp + fp + fn
    f1 = torch.where(denominator > 0.0, 2.0 * tp / denominator, torch.zeros_like(tp))
    overall_denominator = 2.0 * tp.sum() + fp.sum() + fn.sum()
    overall_f1 = (
        float((2.0 * tp.sum() / overall_denominator).item())
        if float(overall_denominator.item()) > 0.0
        else 0.0
    )
    aps = [
        binary_average_precision_tie_stable(logits[:, index], targets[:, index])
        for index in range(logits.shape[1])
    ]
    aucs = [binary_roc_auc(logits[:, index], targets[:, index]) for index in range(logits.shape[1])]
    if any(not math.isfinite(value) for value in (*aps, *aucs)):
        raise ValueError("formal artifact ranking metrics reject undefined per-label AP/AUC")
    bundle = {
        "mF1": float(f1.mean().item()),
        "oF1": overall_f1,
        "mAP": float(sum(aps) / len(aps)),
        "AUC": float(sum(aucs) / len(aucs)),
        "ranking_source": "raw_logits",
        "decision_source": decision_source,
    }
    if not all(
        math.isfinite(float(bundle[key])) for key in ("mF1", "oF1", "mAP", "AUC")
    ):
        raise ValueError("formal artifact metrics must be finite")
    rows = [
        {
            "id": index,
            "name": str(label_names[index]),
            "F1": float(f1[index].item()),
            "AP": float(aps[index]),
            "AUC": float(aucs[index]),
            "support": int(targets[:, index].sum().item()),
            "threshold": float(thresholds[index].item()),
        }
        for index in range(logits.shape[1])
    ]
    return bundle, rows


def _metric_record(action: Mapping[str, Any], reason: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": dict(action),
        "reason": dict(reason),
        "joint": 0.5 * (float(action["mF1"]) + float(reason["mF1"])),
    }


def _calibration_boundaries(calibration: Mapping[str, Any], *, targets: int) -> Tensor:
    chosen = calibration.get("chosen")
    if not isinstance(chosen, Mapping):
        raise ValueError("P15 calibration chosen deployment parameters are missing")
    thresholds = chosen.get("threshold")
    temperatures = chosen.get("temperature")
    if not isinstance(thresholds, list) or not isinstance(temperatures, list):
        raise ValueError("P15 calibration chosen deployment parameters are malformed")
    if len(thresholds) != targets or len(temperatures) != targets:
        raise ValueError("P15 calibration chosen deployment parameter count is invalid")
    boundary = torch.tensor(thresholds, dtype=torch.float64) * torch.tensor(temperatures, dtype=torch.float64)
    if not bool(torch.isfinite(boundary).all()):
        raise ValueError("P15 calibration deployment boundary must be finite")
    return boundary


@torch.no_grad()
def evaluate_rael_test_only(
    *,
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    action_calibration: Mapping[str, Any],
    reason_calibration: Mapping[str, Any],
    expected_train_calib_split_hash: str,
    expected_test_split_hash: str,
    action_schema_path: str | Path,
    reason_schema_path: str | Path,
    device: torch.device,
    case_collector: object | None = None,
    case_export_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate exactly the test split and publish only P18-compatible artifacts."""

    if len(BRANCH_NAMES) != 14 or len(set(BRANCH_NAMES)) != 14:
        raise RuntimeError("model BRANCH_NAMES must define exactly fourteen diagnostic branches")
    branch_names = tuple(BRANCH_NAMES)
    was_training = model.training
    model.eval()
    try:
        case_collector = _require_case_export_collector(
            case_collector, case_export_provenance
        )
        expected_train_calib_split_hash = _require_sha256(
            expected_train_calib_split_hash, context="expected_train_calib_split_hash"
        )
        expected_test_split_hash = _require_sha256(
            expected_test_split_hash, context="expected_test_split_hash"
        )
        if expected_train_calib_split_hash == expected_test_split_hash:
            raise ValueError("train_calib and test split hashes must differ")
        action_names = _load_action_semantic_names(action_schema_path)
        reason_names = _load_reason_semantic_names(reason_schema_path)
        _require_calibration_descriptor(action_calibration, targets=_ACTION_DIM, context="action_calibration")
        _require_calibration_descriptor(reason_calibration, targets=_REASON_DIM, context="reason_calibration")
        # Validate the complete P15 digest-bound object before any DINO encode.
        serialize_calibration_result(action_calibration)
        serialize_calibration_result(reason_calibration)
        _require_calibration_source_hash(
            action_calibration,
            expected=expected_train_calib_split_hash,
            context="action_calibration",
        )
        _require_calibration_source_hash(
            reason_calibration,
            expected=expected_train_calib_split_hash,
            context="reason_calibration",
        )
        action_targets_rows: list[Tensor] = []
        reason_targets_rows: list[Tensor] = []
        branch_rows: dict[str, dict[str, list[Tensor]]] = {
            name: {"action": [], "reason": []} for name in branch_names
        }
        file_names: list[str] = []
        seen_file_names: set[str] = set()
        test_split_hasher = _P12CompactSplitHasher()
        diagnostic_rows: list[dict[str, Any]] = []

        for batch in batches:
            images, action_targets, reason_targets, names = _require_batch(batch)
            if any(name in seen_file_names for name in names):
                raise ValueError("test file_names must be globally unique across batches")
            seen_file_names.update(names)
            for name in names:
                test_split_hasher.add(name)
            action_targets_rows.append(
                _seal_tensor(
                    action_targets,
                    dtype=torch.float32,
                    shape=tuple(action_targets.shape),
                    context="action targets",
                )
            )
            reason_targets_rows.append(
                _seal_tensor(
                    reason_targets,
                    dtype=torch.float32,
                    shape=tuple(reason_targets.shape),
                    context="reason targets",
                )
            )
            file_names.extend(names)
            field = model.encode_images(images.to(device))
            outputs = model.decode_from_field(field, diagnostic_modes=branch_names)
            if not isinstance(outputs, Mapping):
                raise TypeError("RAEL decode_from_field must return a mapping")
            _diagnostic_dino_call_count(outputs)
            branches = outputs.get("branch_logits")
            if (
                not isinstance(branches, Mapping)
                or len(branches) != len(branch_names)
                or set(branches) != set(branch_names)
            ):
                raise ValueError("RAEL evaluator requires exactly the fourteen branch-logit keys")
            final_action = outputs.get("action_logits_final")
            final_reason = outputs.get("reason_logits_final")
            if not isinstance(final_action, Tensor) or not isinstance(final_reason, Tensor):
                raise TypeError("RAEL decode must return final action/reason logits")
            if final_action.shape != action_targets.shape or final_reason.shape != reason_targets.shape:
                raise ValueError("RAEL final logits must match target shapes")
            if not bool(torch.isfinite(final_action).all()) or not bool(torch.isfinite(final_reason).all()):
                raise ValueError("RAEL final logits must be finite")
            # P21 real models expose the complete P18 mechanism surface.
            # Protocol-only evaluator doubles may omit it; the full publisher
            # still fails closed when real diagnostic rows are absent.
            diagnostic_required = {
                "slot_masks", "slot_area", "slot_reliability",
                "layer_weights_action", "layer_weights_reason", "layer_weights_slots",
                "action_unary_contributions", "reason_unary_contributions",
                "action_pairwise_contributions", "reason_pairwise_contributions",
                "action_global_contribution", "reason_global_contribution",
                "named_contribution_ratio", "latent_contribution_ratio",
                "positive_contribution", "negative_contribution", "null_mass",
            }
            if diagnostic_required.issubset(outputs):
                diagnostic_rows.append(_epoch_diagnostic_row(outputs))
            for name in branch_names:
                branch = branches[name]
                if not isinstance(branch, Mapping):
                    raise TypeError(f"branch {name} must be a mapping")
                action_logits = branch.get("action")
                reason_logits = branch.get("reason")
                if not isinstance(action_logits, Tensor) or not isinstance(reason_logits, Tensor):
                    raise TypeError(f"branch {name} logits must be tensors")
                if action_logits.shape != action_targets.shape or reason_logits.shape != reason_targets.shape:
                    raise ValueError(f"branch {name} logits are invalid")
                if not bool(torch.isfinite(action_logits).all()) or not bool(torch.isfinite(reason_logits).all()):
                    raise ValueError(f"branch {name} logits must be finite")
                if name == "full" and (
                    action_logits.device != final_action.device
                    or reason_logits.device != final_reason.device
                    or action_logits.dtype != final_action.dtype
                    or reason_logits.dtype != final_reason.dtype
                    or not torch.equal(action_logits, final_action)
                    or not torch.equal(reason_logits, final_reason)
                ):
                    raise ValueError("full branch logits must equal action_logits_final/reason_logits_final elementwise")
                branch_rows[name]["action"].append(
                    _seal_tensor(action_logits, dtype=torch.float32, shape=tuple(action_targets.shape), context=f"branch {name} action")
                )
                branch_rows[name]["reason"].append(
                    _seal_tensor(reason_logits, dtype=torch.float32, shape=tuple(reason_targets.shape), context=f"branch {name} reason")
                )

            if case_collector is not None:
                case_collector.consume(
                    file_names=names,
                    action_targets=action_targets,
                    reason_targets=reason_targets,
                    outputs=outputs,
                    action_calibration=action_calibration,
                    reason_calibration=reason_calibration,
                    action_names=action_names,
                    reason_names=reason_names,
                )

            # Retain only sealed targets/logits and filenames across batches.
            del (
                batch,
                images,
                action_targets,
                reason_targets,
                names,
                field,
                outputs,
                branches,
                final_action,
                final_reason,
                branch,
                action_logits,
                reason_logits,
            )

        if not action_targets_rows:
            raise ValueError("RAEL evaluator received no test rows")
        if test_split_hasher.hexdigest() != expected_test_split_hash:
            raise ValueError("test file_names split hash does not match expected test split")
        case_exports = (
            None
            if case_collector is None
            else case_collector.finalize(case_export_provenance)
        )
        action_targets = torch.cat(action_targets_rows, dim=0)
        reason_targets = torch.cat(reason_targets_rows, dim=0)
        branch_logits = {
            name: {family: torch.cat(rows[family], dim=0) for family in ("action", "reason")}
            for name, rows in branch_rows.items()
        }
        zero_action = torch.zeros(_ACTION_DIM, dtype=torch.float64)
        zero_reason = torch.zeros(_REASON_DIM, dtype=torch.float64)
        branch_payloads: list[dict[str, Any]] = []
        for name in branch_names:
            action_bundle, action_rows = _family_metrics_and_rows(
                logits=branch_logits[name]["action"],
                targets=action_targets,
                decisions=branch_logits[name]["action"] > 0.0,
                thresholds=zero_action,
                label_names=action_names,
                decision_source="raw_zero_threshold",
            )
            reason_bundle, reason_rows = _family_metrics_and_rows(
                logits=branch_logits[name]["reason"],
                targets=reason_targets,
                decisions=branch_logits[name]["reason"] > 0.0,
                thresholds=zero_reason,
                label_names=reason_names,
                decision_source="raw_zero_threshold",
            )
            branch_payloads.append(
                {
                    "name": name,
                    "config": {"diagnostic_mode": name},
                    "metrics": _metric_record(action_bundle, reason_bundle),
                    "per_action": action_rows,
                    "per_reason": reason_rows,
                }
            )

        raw_action = branch_logits["full"]["action"]
        raw_reason = branch_logits["full"]["reason"]
        raw_action_bundle, _ = _family_metrics_and_rows(
            logits=raw_action,
            targets=action_targets,
            decisions=raw_action > 0.0,
            thresholds=zero_action,
            label_names=action_names,
            decision_source="raw_zero_threshold",
        )
        raw_reason_bundle, _ = _family_metrics_and_rows(
            logits=raw_reason,
            targets=reason_targets,
            decisions=raw_reason > 0.0,
            thresholds=zero_reason,
            label_names=reason_names,
            decision_source="raw_zero_threshold",
        )
        raw_metrics = {"metrics": _metric_record(raw_action_bundle, raw_reason_bundle)}

        action_deploy = apply_posthoc_calibration(raw_action, action_calibration)
        reason_deploy = apply_posthoc_calibration(raw_reason, reason_calibration)
        action_boundary = _calibration_boundaries(action_calibration, targets=_ACTION_DIM)
        reason_boundary = _calibration_boundaries(reason_calibration, targets=_REASON_DIM)
        deploy_action_bundle, deploy_action_rows = _family_metrics_and_rows(
            logits=raw_action,
            targets=action_targets,
            decisions=action_deploy["decision"],
            thresholds=action_boundary,
            label_names=action_names,
            decision_source="p15_train_calib_posthoc",
        )
        deploy_reason_bundle, deploy_reason_rows = _family_metrics_and_rows(
            logits=raw_reason,
            targets=reason_targets,
            decisions=reason_deploy["decision"],
            thresholds=reason_boundary,
            label_names=reason_names,
            decision_source="p15_train_calib_posthoc",
        )
        deploy_metrics = {"metrics": _metric_record(deploy_action_bundle, deploy_reason_bundle)}
        raw_action_tensor = _seal_tensor(raw_action, dtype=torch.float32, shape=(action_targets.shape[0], _ACTION_DIM), context="raw action logits")
        raw_reason_tensor = _seal_tensor(raw_reason, dtype=torch.float32, shape=(reason_targets.shape[0], _REASON_DIM), context="raw reason logits")
        action_margin = _seal_tensor(
            action_deploy["diagnostic_margin"],
            dtype=torch.float32,
            shape=(action_targets.shape[0], _ACTION_DIM),
            context="deploy action diagnostic margin",
        )
        reason_margin = _seal_tensor(
            reason_deploy["diagnostic_margin"],
            dtype=torch.float32,
            shape=(reason_targets.shape[0], _REASON_DIM),
            context="deploy reason diagnostic margin",
        )

        return {
            "raw_metrics": raw_metrics,
            "deploy_metrics": deploy_metrics,
            "branch_metrics": {"branches": branch_payloads},
            "per_action": {"rows": deploy_action_rows},
            "per_reason": {"rows": deploy_reason_rows},
            "selection": {
                "split": "test",
                "metric": "deploy_fixed_joint",
                "internal_test_selected": True,
                "publication_eligible": False,
            },
            "primary_best_value": deploy_metrics["metrics"]["joint"],
            "file_names": list(file_names),
            "case_exports": case_exports,
            "diagnostic_rows": diagnostic_rows,
            "tensors": {
                "logits_raw": {"action": raw_action_tensor, "reason": raw_reason_tensor},
                "logits_deploy": {"action": action_margin, "reason": reason_margin},
                "labels": {"action": action_targets, "reason": reason_targets},
            },
        }
    finally:
        model.train(was_training)


__all__ = [
    "binary_average_precision_tie_stable",
    "binary_roc_auc",
    "evaluate_rael_test_only",
]
