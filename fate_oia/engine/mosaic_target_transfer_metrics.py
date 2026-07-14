"""Counterfactual evidence-transfer metrics for MOSAIC action and reason targets."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Sequence

import torch


_DIRECTIONS = {"support": 1.0, "veto": -1.0}
_SCHEMA_VERSION = "mosaic_target_transfer.v1"


@dataclass(frozen=True)
class TargetTransferInputs:
    """Required, aligned outputs of selected, random, and deletion interventions.

    Target probabilities have shape ``[sample, factor, target]``.  The two masks
    are persisted visual-audit masks; equal factor cardinality makes the random
    intervention an equal-mass control rather than an unavailable proxy.
    """

    factor_ids: Sequence[str]
    target_ids: Sequence[str]
    directions: Sequence[Sequence[str]]
    factor_visual_evidence: Any
    selected_factor_mask: Any
    matched_random_factor_mask: Any
    target_evaluation_mask: Any
    target_labels: Any
    selected_target_prob: Any
    matched_random_target_prob: Any
    deleted_target_prob: Any
    evidence_threshold: float = 0.0


def _tolist(value: Any) -> Any:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "tolist") and callable(value.tolist):
        value = value.tolist()
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    value = _tolist(value)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"target transfer {name} must be a non-string sequence")
    return list(value)


def _matrix(value: Any, rows: int, columns: int, name: str) -> list[list[Any]]:
    matrix = _sequence(value, name)
    if len(matrix) != rows:
        raise ValueError(f"target transfer {name} must have {rows} sample rows")
    result = [_sequence(row, name) for row in matrix]
    if any(len(row) != columns for row in result):
        raise ValueError(f"target transfer {name} must have {columns} columns")
    return result


def _cube(value: Any, rows: int, factors: int, targets: int, name: str) -> list[list[list[float]]]:
    cube = _sequence(value, name)
    if len(cube) != rows:
        raise ValueError(f"target transfer {name} must have {rows} sample rows")
    result: list[list[list[float]]] = []
    for sample in cube:
        factor_rows = _sequence(sample, name)
        if len(factor_rows) != factors:
            raise ValueError(f"target transfer {name} must have {factors} factor columns")
        target_rows = [_sequence(row, name) for row in factor_rows]
        if any(len(row) != targets for row in target_rows):
            raise ValueError(f"target transfer {name} must have {targets} target columns")
        result.append([[_probability(item, name) for item in target_row] for target_row in target_rows])
    return result


def _probability(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"target transfer {name} must contain numeric probabilities") from error
    if not isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"target transfer {name} probabilities must be finite values in [0, 1]")
    return number


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("target transfer counterfactual has no eligible observations")
    return sum(values) / len(values)


def _average_precision(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("target transfer AP requires aligned non-empty scores and labels")
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError("target transfer AP requires positive and negative target examples")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for index, (_, positive) in enumerate(ordered, start=1):
        if positive:
            true_positives += 1
            precision_sum += true_positives / index
    return precision_sum / positives


def _validate_ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    ids = tuple(str(value) for value in values)
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"target transfer requires unique non-empty {name}")
    return ids


def compute_target_transfer_metrics(inputs: TargetTransferInputs) -> dict[str, Any]:
    """Measure whether visual factor evidence transfers to every action/reason target.

    ``selected_target_prob``, ``matched_random_target_prob``, and
    ``deleted_target_prob`` must come from three actual model interventions.  This
    function only measures them; it never fabricates a zero or duplicates a branch.
    """

    factor_ids = _validate_ids(inputs.factor_ids, "factor_ids")
    target_ids = _validate_ids(inputs.target_ids, "target_ids")
    factor_count, target_count = len(factor_ids), len(target_ids)
    evidence_rows = _sequence(inputs.factor_visual_evidence, "factor_visual_evidence")
    if not evidence_rows:
        raise ValueError("target transfer requires at least one sample")
    sample_count = len(evidence_rows)
    visual_evidence = [
        [_probability(value, "factor_visual_evidence") for value in row]
        for row in _matrix(evidence_rows, sample_count, factor_count, "factor_visual_evidence")
    ]
    if not isfinite(float(inputs.evidence_threshold)) or not 0.0 <= float(inputs.evidence_threshold) < 1.0:
        raise ValueError("target transfer evidence_threshold must be finite and in [0, 1)")
    directions = _matrix(inputs.directions, factor_count, target_count, "directions")
    if any(direction not in {*_DIRECTIONS, "none"} for row in directions for direction in row):
        raise ValueError("target transfer directions must be support, veto, or none")
    selected_mask = _matrix(inputs.selected_factor_mask, sample_count, factor_count, "selected_factor_mask")
    random_mask = _matrix(inputs.matched_random_factor_mask, sample_count, factor_count, "matched_random_factor_mask")
    evaluation_mask = _matrix(inputs.target_evaluation_mask, sample_count, target_count, "target_evaluation_mask")
    labels = [
        [_probability(value, "target_labels") for value in row]
        for row in _matrix(inputs.target_labels, sample_count, target_count, "target_labels")
    ]
    if any(value not in (0.0, 1.0) for row in labels for value in row):
        raise ValueError("target transfer target_labels must be binary 0/1 observations")
    selected = _cube(inputs.selected_target_prob, sample_count, factor_count, target_count, "selected_target_prob")
    random = _cube(inputs.matched_random_target_prob, sample_count, factor_count, target_count, "matched_random_target_prob")
    deleted = _cube(inputs.deleted_target_prob, sample_count, factor_count, target_count, "deleted_target_prob")

    selected_counts = [sum(bool(row[factor]) for row in selected_mask) for factor in range(factor_count)]
    random_counts = [sum(bool(row[factor]) for row in random_mask) for factor in range(factor_count)]
    if selected_counts != random_counts or any(count == 0 for count in selected_counts):
        raise ValueError("target transfer requires non-empty matched random masks with equal factor mass")

    per_target: list[dict[str, Any]] = []
    threshold = float(inputs.evidence_threshold)
    for factor_index, factor_id in enumerate(factor_ids):
        for target_index, target_id in enumerate(target_ids):
            if directions[factor_index][target_index] == "none":
                continue
            rows = [
                sample_index
                for sample_index in range(sample_count)
                if bool(selected_mask[sample_index][factor_index])
                and bool(evaluation_mask[sample_index][target_index])
                and visual_evidence[sample_index][factor_index] > threshold
            ]
            if not rows:
                raise ValueError(f"target transfer {factor_id}->{target_id} has no selected visual evidence rows")
            target_labels = [bool(labels[row][target_index]) for row in rows]
            selected_scores = [selected[row][factor_index][target_index] for row in rows]
            deleted_scores = [deleted[row][factor_index][target_index] for row in rows]
            random_scores = [random[row][factor_index][target_index] for row in rows]
            selected_ap = _average_precision(selected_scores, target_labels)
            deleted_ap = _average_precision(deleted_scores, target_labels)
            direction = directions[factor_index][target_index]
            direction_rows = [label if direction == "support" else not label for label in target_labels]
            if not any(direction_rows):
                raise ValueError(f"target transfer {factor_id}->{target_id} has no direction-specific rows")
            sign = _DIRECTIONS[direction]
            signed_selected = [sign * (on - off) for on, off in zip(selected_scores, deleted_scores)]
            # Both effects share the same full-evidence baseline. This makes
            # TES answer whether deleting the selected factor hurts the target
            # more than deleting an equal-mass random factor.
            signed_random = [sign * (on - control) for on, control in zip(selected_scores, random_scores)]
            selected_effect = _mean([effect for effect, keep in zip(signed_selected, direction_rows) if keep])
            random_effect = _mean([effect for effect, keep in zip(signed_random, direction_rows) if keep])
            calibration_selected = _mean([1.0 - abs(score - label) for score, label in zip(selected_scores, target_labels)])
            calibration_deleted = _mean([1.0 - abs(score - label) for score, label in zip(deleted_scores, target_labels)])
            tet = selected_effect
            tes = tet - random_effect
            cca = calibration_selected - calibration_deleted
            ap_delta = selected_ap - deleted_ap
            per_target.append(
                {
                    "factor_id": factor_id,
                    "target_id": target_id,
                    "direction": direction,
                    "n": len(rows),
                    "selected_effect": selected_effect,
                    "matched_random_effect": random_effect,
                    "signed_effect": tet,
                    "tet": tet,
                    "tes": tes,
                    "cca": cca,
                    "ap_delta": ap_delta,
                    "selected_ap": selected_ap,
                    "deleted_ap": deleted_ap,
                    "visual_evidence_mean": _mean([visual_evidence[row][factor_index] for row in rows]),
                    "admitted": tes > 0.0 and cca > 0.0 and ap_delta > 0.0,
                }
            )
    if not per_target:
        raise ValueError("target transfer has no candidate factor-target pairs")
    summary = {
        "pair_count": len(per_target),
        "mean_selected_effect": _mean([item["selected_effect"] for item in per_target]),
        "mean_matched_random_effect": _mean([item["matched_random_effect"] for item in per_target]),
        "mean_tet": _mean([item["tet"] for item in per_target]),
        "mean_tes": _mean([item["tes"] for item in per_target]),
        "mean_cca": _mean([item["cca"] for item in per_target]),
        "mean_ap_delta": _mean([item["ap_delta"] for item in per_target]),
        "admitted_rate": _mean([1.0 if item["admitted"] else 0.0 for item in per_target]),
    }
    return {"schema_version": _SCHEMA_VERSION, "per_target": per_target, "summary": summary}


@torch.no_grad()
def collect_target_transfer_metrics(
    model: torch.nn.Module,
    loader: Any,
    *,
    factor_ids: Sequence[str],
    target_ids: Sequence[str],
    directions: Sequence[Sequence[str]],
    target_kind: str,
    device: torch.device,
    route_mode: str,
    latent_enabled: bool,
) -> dict[str, Any]:
    """Run full, selected-factor deletion, and matched random deletion forwards.

    This collector is audit-only and label-free in every model forward. Labels
    enter only after logits have been produced, when transfer effects are scored.
    """
    if target_kind not in {"action", "reason"}:
        raise ValueError("target transfer target_kind must be action or reason")
    factor_count, target_count = len(factor_ids), len(target_ids)
    probability_key = "action_final_logits" if target_kind == "action" else "reason_observed_logits"
    label_key = "action" if target_kind == "action" else "reason"
    visual_evidence: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    full_probability: list[torch.Tensor] = []
    deleted_probability: list[torch.Tensor] = []
    random_deleted_probability: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            splits = batch.get("split")
            split_values = [splits] if isinstance(splits, str) else list(splits or [])
            if not split_values or any(value != "train_audit" for value in split_values):
                raise ValueError("target transfer collector accepts train_audit only")
            images = batch["image"].to(device)
            target = batch[label_key].to(device).float()
            if target.shape != (images.shape[0], target_count):
                raise ValueError("target transfer labels do not match target_ids")
            full = model(
                images, route_mode=route_mode, latent_enabled=latent_enabled,
                return_masks=False,
            )
            full_prob = torch.sigmoid(full[probability_key]).float()
            evidence = full["factor_presence_prob"].float()
            if evidence.shape != (images.shape[0], factor_count):
                raise ValueError("target transfer factor evidence does not match factor_ids")
            deleted_arms: list[torch.Tensor] = []
            random_arms: list[torch.Tensor] = []
            for factor_index in range(factor_count):
                keep = torch.ones(images.shape[0], factor_count, device=device)
                keep[:, factor_index] = 0.0
                deleted = model(
                    images, route_mode=route_mode, latent_enabled=latent_enabled,
                    return_masks=False, factor_intervention_keep_mask=keep,
                )
                random_keep = torch.ones_like(keep)
                random_keep[:, (factor_index + 1) % factor_count] = 0.0
                random_deleted = model(
                    images, route_mode=route_mode, latent_enabled=latent_enabled,
                    return_masks=False, factor_intervention_keep_mask=random_keep,
                )
                deleted_arms.append(torch.sigmoid(deleted[probability_key]).float())
                random_arms.append(torch.sigmoid(random_deleted[probability_key]).float())
            visual_evidence.append(evidence.cpu())
            labels.append(target.cpu())
            full_probability.append(full_prob[:, None, :].expand(-1, factor_count, -1).cpu())
            deleted_probability.append(torch.stack(deleted_arms, dim=1).cpu())
            random_deleted_probability.append(torch.stack(random_arms, dim=1).cpu())
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("target transfer collector received no train_audit rows")
    evidence = torch.cat(visual_evidence)
    label = torch.cat(labels)
    selected = torch.cat(full_probability)
    deleted = torch.cat(deleted_probability)
    random_deleted = torch.cat(random_deleted_probability)
    sample_count = label.shape[0]
    return compute_target_transfer_metrics(TargetTransferInputs(
        factor_ids=factor_ids,
        target_ids=target_ids,
        directions=directions,
        factor_visual_evidence=evidence,
        selected_factor_mask=torch.ones(sample_count, factor_count, dtype=torch.bool),
        matched_random_factor_mask=torch.ones(sample_count, factor_count, dtype=torch.bool),
        target_evaluation_mask=torch.ones(sample_count, target_count, dtype=torch.bool),
        target_labels=label,
        selected_target_prob=selected,
        matched_random_target_prob=random_deleted,
        deleted_target_prob=deleted,
    ))


@torch.no_grad()
def collect_joint_target_transfer_metrics(
    model: torch.nn.Module,
    loader: Any,
    *,
    factor_ids: Sequence[str],
    action_ids: Sequence[str],
    reason_ids: Sequence[str],
    action_directions: Sequence[Sequence[str]],
    reason_directions: Sequence[Sequence[str]],
    device: torch.device,
    route_mode: str,
    latent_enabled: bool,
) -> dict[str, Any]:
    """Collect action and reason transfer in one intervention sweep."""
    factor_count = len(factor_ids)
    target_ids = tuple(f"action:{name}" for name in action_ids) + tuple(f"reason:{name}" for name in reason_ids)
    directions = tuple(
        tuple(action_directions[factor]) + tuple(reason_directions[factor])
        for factor in range(factor_count)
    )
    visual_evidence: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    full_probability: list[torch.Tensor] = []
    deleted_probability: list[torch.Tensor] = []
    random_deleted_probability: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            splits = batch.get("split")
            split_values = [splits] if isinstance(splits, str) else list(splits or [])
            if not split_values or any(value != "train_audit" for value in split_values):
                raise ValueError("target transfer collector accepts train_audit only")
            images = batch["image"].to(device)
            audit_field = model.dino(images) if hasattr(model, "dino") else None
            field_kwargs = {"precomputed_dino_field": audit_field} if audit_field is not None else {}
            target = torch.cat((batch["action"], batch["reason"]), dim=1).to(device).float()
            if target.shape != (images.shape[0], len(target_ids)):
                raise ValueError("joint target transfer labels do not match action/reason ids")
            full = model(
                images, route_mode=route_mode, latent_enabled=latent_enabled,
                return_masks=False, **field_kwargs,
            )
            full_prob = torch.cat((
                torch.sigmoid(full["action_final_logits"]),
                torch.sigmoid(full["reason_observed_logits"]),
            ), dim=1).float()
            evidence = full["factor_presence_prob"].float()
            if evidence.shape != (images.shape[0], factor_count):
                raise ValueError("joint target transfer factor evidence does not match factor ids")
            deleted_arms: list[torch.Tensor] = []
            random_arms: list[torch.Tensor] = []
            for factor_index in range(factor_count):
                keep = torch.ones(images.shape[0], factor_count, device=device)
                keep[:, factor_index] = 0.0
                random_keep = torch.ones_like(keep)
                random_keep[:, (factor_index + 1) % factor_count] = 0.0
                arms = []
                for intervention in (keep, random_keep):
                    output = model(
                        images, route_mode=route_mode, latent_enabled=latent_enabled,
                        return_masks=False, factor_intervention_keep_mask=intervention,
                        **field_kwargs,
                    )
                    arms.append(torch.cat((
                        torch.sigmoid(output["action_final_logits"]),
                        torch.sigmoid(output["reason_observed_logits"]),
                    ), dim=1).float())
                deleted_arms.append(arms[0])
                random_arms.append(arms[1])
            visual_evidence.append(evidence.cpu())
            labels.append(target.cpu())
            full_probability.append(full_prob[:, None, :].expand(-1, factor_count, -1).cpu())
            deleted_probability.append(torch.stack(deleted_arms, dim=1).cpu())
            random_deleted_probability.append(torch.stack(random_arms, dim=1).cpu())
    finally:
        model.train(was_training)
    if not labels:
        raise ValueError("joint target transfer collector received no train_audit rows")
    label = torch.cat(labels)
    sample_count = label.shape[0]
    return compute_target_transfer_metrics(TargetTransferInputs(
        factor_ids=factor_ids,
        target_ids=target_ids,
        directions=directions,
        factor_visual_evidence=torch.cat(visual_evidence),
        selected_factor_mask=torch.ones(sample_count, factor_count, dtype=torch.bool),
        matched_random_factor_mask=torch.ones(sample_count, factor_count, dtype=torch.bool),
        target_evaluation_mask=torch.ones(sample_count, len(target_ids), dtype=torch.bool),
        target_labels=label,
        selected_target_prob=torch.cat(full_probability),
        matched_random_target_prob=torch.cat(random_deleted_probability),
        deleted_target_prob=torch.cat(deleted_probability),
    ))
