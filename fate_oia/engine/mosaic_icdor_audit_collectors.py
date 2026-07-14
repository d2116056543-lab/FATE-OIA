from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


_ABLATION_MODES = ("full", "content_only", "prior_only", "query_shuffled", "image_shuffled")
_DIRECTION_INDEX = {"support": 0, "veto": 1}


def _split_values(batch: Mapping[str, Any]) -> list[str]:
    split = batch.get("split")
    values = [split] if isinstance(split, str) else list(split) if isinstance(split, Sequence) else []
    if not values or any(value != "train_audit" for value in values):
        raise ValueError("IC-DOR audit collectors accept train_audit batches only")
    return values


def _images(batch: Mapping[str, Any], device: torch.device) -> torch.Tensor:
    image = batch.get("image")
    if not isinstance(image, torch.Tensor) or image.ndim < 2:
        raise ValueError("IC-DOR audit batch requires an image tensor")
    return image.to(device)


def _records(batch: Mapping[str, Any], count: int) -> Sequence[dict[str, Any] | None]:
    for key in ("grounding_records", "bdd100k_records", "raw_records"):
        value = batch.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == count:
            return value
    raise ValueError("IC-DOR audit batch requires BDD100K grounding_records aligned with image rows")


def _forward(model: torch.nn.Module, images: torch.Tensor, mode: str, forward_kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    output = model(images, factor_ablation_mode=mode, **dict(forward_kwargs))
    if not isinstance(output, Mapping):
        raise ValueError("IC-DOR audit model forward must return a mapping")
    return output


def _tensor(output: Mapping[str, Any], key: str, *, rows: int, columns: int | None = None) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor) or value.shape[0] != rows:
        raise ValueError(f"IC-DOR audit forward requires {key} aligned with batch rows")
    if columns is not None and (value.ndim < 2 or value.shape[1] != columns):
        raise ValueError(f"IC-DOR audit forward requires {key} with the factor dimension")
    return value.detach().float().cpu()


def _average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    targets = targets.bool()
    positive_count = int(targets.sum().item())
    if positive_count == 0 or positive_count == targets.numel():
        raise ValueError("IC-DOR audit metric requires both confirmed positives and reliable negatives")
    order = torch.argsort(scores, descending=True, stable=True)
    ordered = targets.index_select(0, order).float()
    precision = ordered.cumsum(0) / torch.arange(1, ordered.numel() + 1, dtype=torch.float32)
    return float((precision * ordered).sum().item() / positive_count)


def _expected_calibration_error(probability: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    if probability.numel() == 0:
        raise ValueError("IC-DOR ECE requires observed factor targets")
    total = float(probability.numel())
    error = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        keep = (probability >= lower) & ((probability < upper) if index + 1 < bins else (probability <= upper))
        if bool(keep.any()):
            error += float(keep.sum().item()) / total * abs(float(probability[keep].mean()) - float(target[keep].mean()))
    return error


def _derangement(length: int, generator: torch.Generator) -> torch.Tensor:
    if length < 2:
        raise ValueError("IC-DOR grounding-minus-random requires at least two geometry observations")
    # A cyclic shift is a genuine random pairing while guaranteeing no row keeps its own geometry.
    shift = int(torch.randint(1, length, (1,), generator=generator).item())
    return torch.roll(torch.arange(length), shifts=shift)


def _grounding_delta(mask: torch.Tensor, geometry: torch.Tensor, valid: torch.Tensor, generator: torch.Generator) -> float:
    rows = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if rows.numel() == 0:
        # Both observed overlap sums are exactly empty; this is an observed zero, not a placeholder.
        return 0.0
    selected_mask = mask.index_select(0, rows).clamp_min(0.0)
    selected_geometry = geometry.index_select(0, rows).clamp_min(0.0)
    aligned = (selected_mask * selected_geometry).sum((-2, -1)) / selected_mask.sum((-2, -1)).clamp_min(1e-12)
    random_geometry = selected_geometry.index_select(0, _derangement(int(rows.numel()), generator))
    random = (selected_mask * random_geometry).sum((-2, -1)) / selected_mask.sum((-2, -1)).clamp_min(1e-12)
    return float((aligned - random).mean())


def _quantile(values: list[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


def _stratified_indices(target: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    positive = torch.nonzero(target.bool(), as_tuple=False).flatten()
    negative = torch.nonzero(~target.bool(), as_tuple=False).flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("IC-DOR bootstrap requires both confirmed positives and reliable negatives")
    return torch.cat((
        positive.index_select(0, torch.randint(positive.numel(), (positive.numel(),), generator=generator)),
        negative.index_select(0, torch.randint(negative.numel(), (negative.numel(),), generator=generator)),
    ))


def _prototype_stats(weights: torch.Tensor) -> tuple[float, float, int]:
    if weights.ndim != 2:
        raise ValueError("IC-DOR audit requires per-example prototype_weights")
    occupancy = weights.mean(0)
    valid = occupancy > 0.0
    if not bool(valid.any()):
        raise ValueError("IC-DOR audit observed no valid prototypes")
    normalized = occupancy[valid] / occupancy[valid].sum().clamp_min(1e-12)
    effective = float((-(normalized * normalized.clamp_min(1e-12).log()).sum()).exp())
    dominant = float((weights.max(-1).values > 0.85).float().mean())
    dead = int((occupancy <= 1e-4).sum().item())
    return effective, dominant, dead


@torch.no_grad()
def collect_factor_audit(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    grounding_builder: Any,
    *,
    factor_names: Sequence[str],
    device: torch.device,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 0,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect certificate-ready IC-DOR factor statistics from train_audit only.

    ``audit_views`` must contain at least two BDD100K views registered to the base
    image coordinate system.  The collector never synthesizes views or labels.
    """
    if not factor_names or len(set(factor_names)) != len(factor_names):
        raise ValueError("IC-DOR factor audit requires unique factor names")
    if bootstrap_replicates < 1:
        raise ValueError("IC-DOR factor audit requires positive bootstrap_replicates")
    kwargs = dict(forward_kwargs or {})
    factor_count = len(factor_names)
    collected: dict[str, list[torch.Tensor]] = {
        "target": [], "known": [], "weak": [], "geometry_known": [], "geometry": [],
        "full": [], "content_only": [], "prior_only": [], "query_shuffled": [], "image_shuffled": [],
        "full_mask": [], "view_mask": [], "mirror_mask": [], "visibility": [], "prototype_weights": [],
    }
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise ValueError("IC-DOR audit loader must yield mapping batches")
            splits = _split_values(batch)
            images = _images(batch, device)
            if len(splits) != images.shape[0]:
                raise ValueError("IC-DOR audit split rows must align with images")
            grounding = grounding_builder(_records(batch, images.shape[0]), device=device, split="train")
            if not isinstance(grounding, Mapping):
                raise ValueError("IC-DOR grounding builder must return a mapping")
            target = _tensor(grounding, "presence_target", rows=images.shape[0], columns=factor_count)
            known = _tensor(grounding, "presence_known_mask", rows=images.shape[0], columns=factor_count).bool()
            weak = _tensor(grounding, "weak_negative_mask", rows=images.shape[0], columns=factor_count).bool()
            geometry_known = _tensor(grounding, "geometry_known_mask", rows=images.shape[0], columns=factor_count).bool()
            geometry = _tensor(grounding, "geometry_masks", rows=images.shape[0], columns=factor_count)
            outputs = {mode: _forward(model, images, mode, kwargs) for mode in _ABLATION_MODES}
            full = outputs["full"]
            views = batch.get("audit_views")
            if not isinstance(views, torch.Tensor) or views.ndim != images.ndim + 1 or views.shape[0] != images.shape[0] or views.shape[1] < 2:
                raise ValueError("IC-DOR factor audit requires at least two registered audit_views per image")
            mirror_view = batch.get("audit_mirror_view")
            if not isinstance(mirror_view, torch.Tensor) or mirror_view.shape != images.shape:
                raise ValueError("IC-DOR factor audit requires a registered audit_mirror_view per image")
            view_output = _forward(model, views[:, 1].to(device), "full", kwargs)
            mirror_output = _forward(model, mirror_view.to(device), "full", kwargs)
            collected["target"].append(target)
            collected["known"].append(known)
            collected["weak"].append(weak)
            collected["geometry_known"].append(geometry_known)
            collected["geometry"].append(geometry)
            for mode, output in outputs.items():
                collected[mode].append(_tensor(output, "factor_presence_prob", rows=images.shape[0], columns=factor_count))
            collected["full_mask"].append(_tensor(full, "factor_soft_masks", rows=images.shape[0], columns=factor_count))
            collected["view_mask"].append(_tensor(view_output, "factor_soft_masks", rows=images.shape[0], columns=factor_count))
            # Mirror views are registered back to the base image before any
            # equivariance score is computed.
            collected["mirror_mask"].append(
                torch.flip(
                    _tensor(mirror_output, "factor_soft_masks", rows=images.shape[0], columns=factor_count),
                    dims=(-1,),
                )
            )
            collected["visibility"].append(_tensor(full, "factor_visibility_prob", rows=images.shape[0], columns=factor_count))
            collected["prototype_weights"].append(_tensor(full, "prototype_weights", rows=images.shape[0], columns=factor_count))
    finally:
        model.train(was_training)
    if not collected["target"]:
        raise ValueError("IC-DOR factor audit loader produced no train_audit observations")
    values = {key: torch.cat(items, dim=0) for key, items in collected.items()}
    generator = torch.Generator().manual_seed(bootstrap_seed)
    factor_stats: dict[str, dict[str, Any]] = {}
    for column, name in enumerate(factor_names):
        target, known, weak = values["target"][:, column], values["known"][:, column], values["weak"][:, column]
        reliable_negative = known & (target <= 0.5)
        confirmed_positive = known & (target > 0.5)
        metric_rows = confirmed_positive | reliable_negative
        full = values["full"][:, column]
        prior = values["prior_only"][:, column]
        content = values["content_only"][:, column]
        query = values["query_shuffled"][:, column]
        image = values["image_shuffled"][:, column]
        if not bool(metric_rows.any()):
            raise ValueError(f"IC-DOR factor {name} has no confirmed audit labels")
        target_metric = confirmed_positive[metric_rows]
        score_full = _average_precision(full[metric_rows], target_metric)
        score_prior = _average_precision(prior[metric_rows], target_metric)
        score_content = _average_precision(content[metric_rows], target_metric)
        score_query = _average_precision(query[metric_rows], target_metric)
        score_image = _average_precision(image[metric_rows], target_metric)
        geometry_valid = values["geometry_known"][:, column]
        grounding_delta = _grounding_delta(values["full_mask"][:, column], values["geometry"][:, column], geometry_valid, generator)
        mask_difference = (values["full_mask"][:, column] - values["view_mask"][:, column]).abs()
        view_consistency = float((1.0 - mask_difference.mean()).clamp(0.0, 1.0))
        mirror_difference = (values["full_mask"][:, column] - values["mirror_mask"][:, column]).abs()
        mirror_consistency = float((1.0 - mirror_difference.mean()).clamp(0.0, 1.0))
        effective, dominant, dead = _prototype_stats(values["prototype_weights"][:, column])
        replicate: dict[str, list[float]] = {
            "full_minus_prior_only": [], "query_shuffle_drop": [], "image_shuffle_drop": [], "grounding_minus_random": [],
        }
        metric_indices = torch.nonzero(metric_rows, as_tuple=False).flatten()
        for _ in range(bootstrap_replicates):
            local = _stratified_indices(target_metric, generator)
            sample = metric_indices.index_select(0, local)
            replicate["full_minus_prior_only"].append(
                _average_precision(full[sample], confirmed_positive[sample]) - _average_precision(prior[sample], confirmed_positive[sample])
            )
            replicate["query_shuffle_drop"].append(
                _average_precision(full[sample], confirmed_positive[sample]) - _average_precision(query[sample], confirmed_positive[sample])
            )
            replicate["image_shuffle_drop"].append(
                _average_precision(full[sample], confirmed_positive[sample]) - _average_precision(image[sample], confirmed_positive[sample])
            )
            geometry_sample = torch.nonzero(geometry_valid, as_tuple=False).flatten()
            if geometry_sample.numel() == 0:
                replicate["grounding_minus_random"].append(0.0)
            else:
                draw = geometry_sample.index_select(0, torch.randint(geometry_sample.numel(), (geometry_sample.numel(),), generator=generator))
                replicate["grounding_minus_random"].append(
                    _grounding_delta(values["full_mask"][draw, column], values["geometry"][draw, column], torch.ones(draw.numel(), dtype=torch.bool), generator)
                )
        factor_stats[str(name)] = {
            "counts": {
                "confirmed_positive": int(confirmed_positive.sum()),
                "reliable_negative": int(reliable_negative.sum()),
                "weak_negative": int((weak & ~known).sum()),
                "unknown": int((~known & ~weak).sum()),
                "geometry_valid": int(geometry_valid.sum()),
            },
            "scores": {
                "full": score_full,
                "content_only": score_content,
                "prior_only": score_prior,
                "query_shuffle_drop": score_full - score_query,
                "image_shuffle_drop": score_full - score_image,
                "grounding_minus_random": grounding_delta,
                "view_consistency": view_consistency,
                "mirror_consistency": mirror_consistency,
                "ece": _expected_calibration_error(full[metric_rows], target_metric.float()),
                "presence_variance": float(full[metric_rows].var(unbiased=False)),
                "visibility_variance": float(values["visibility"][:, column].var(unbiased=False)),
            },
            "prototype": {"effective_count": effective, "dominant_rate": dominant, "dead_count": dead},
            "bootstrap_lcb95": {key: _quantile(samples, 0.05) for key, samples in replicate.items()},
        }
    return {"source_split": "train_audit", "factor_stats": factor_stats}


def _edge_metrics(
    on: torch.Tensor,
    off: torch.Tensor,
    random: torch.Tensor,
    labels: torch.Tensor,
    *,
    direction: str,
) -> dict[str, float]:
    probability_on, probability_off, probability_random = on.sigmoid(), off.sigmoid(), random.sigmoid()
    positive = labels > 0.5
    if not bool(positive.any()) or bool(positive.all()):
        raise ValueError("IC-DOR edge audit requires positive and negative action examples")
    if direction == "support":
        evaluation_rows = positive
        signed = probability_on - probability_off
        random_signed = probability_random - probability_off
    elif direction == "veto":
        evaluation_rows = ~positive
        signed = probability_off - probability_on
        random_signed = probability_off - probability_random
    else:
        raise ValueError("IC-DOR edge direction must be support or veto")
    if not bool(evaluation_rows.any()):
        raise ValueError("IC-DOR edge audit lacks direction-specific evaluation rows")
    tet = float(signed[evaluation_rows].mean())
    random_effect = float(random_signed[evaluation_rows].mean())
    calibration_on = 1.0 - (probability_on - labels).abs()
    calibration_off = 1.0 - (probability_off - labels).abs()
    isolated_edge_ap = _average_precision(probability_on, positive)
    visual_ap = _average_precision(probability_off, positive)
    return {
        "signed_effect": tet,
        "tet": tet,
        "tes": tet - random_effect,
        "cca": float((calibration_on - calibration_off).mean()),
        "ap_delta": isolated_edge_ap - visual_ap,
        "isolated_edge_ap": isolated_edge_ap,
        "visual_ap": visual_ap,
    }


def _edge_key(spec: Mapping[str, Any]) -> str:
    return f"{spec['direction']}:{spec['factor']}->{spec['action']}"


@torch.no_grad()
def collect_edge_intervention_audit(
    model: torch.nn.Module,
    loader: Iterable[Mapping[str, Any]],
    *,
    factor_names: Sequence[str],
    action_names: Sequence[str],
    edge_specs: Sequence[Mapping[str, Any]],
    device: torch.device,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 0,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run matched on/off/equal-mass-random edge interventions on train_audit."""
    if not edge_specs or bootstrap_replicates < 1:
        raise ValueError("IC-DOR edge audit requires edges and positive bootstrap_replicates")
    router = getattr(model, "action_router", None)
    candidate = getattr(router, "candidate_edge_mask", None)
    current = getattr(router, "edge_admission_mask", None)
    setter = getattr(model, "set_edge_admission", None)
    if not isinstance(candidate, torch.Tensor) or not isinstance(current, torch.Tensor) or not callable(setter):
        raise ValueError("IC-DOR edge audit requires a model with candidate and mutable edge admission masks")
    if candidate.shape != current.shape or candidate.ndim != 3 or candidate.shape[0] != 2:
        raise ValueError("IC-DOR edge admission masks must be [support_or_veto, factor, action]")
    if tuple(candidate.shape[1:]) != (len(factor_names), len(action_names)):
        raise ValueError("IC-DOR edge masks do not match factor/action names")
    factor_index, action_index = {name: index for index, name in enumerate(factor_names)}, {name: index for index, name in enumerate(action_names)}
    plans: list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, int]] = []
    base = current.detach().clone()
    for spec in edge_specs:
        if not isinstance(spec, Mapping) or set(("factor", "action", "direction", "polarity")) - set(spec):
            raise ValueError("IC-DOR edge specs require factor/action/direction/polarity")
        factor, action, direction = str(spec["factor"]), str(spec["action"]), str(spec["direction"])
        if factor not in factor_index or action not in action_index or direction not in _DIRECTION_INDEX:
            raise ValueError("IC-DOR edge spec is not in the factor/action ontology")
        direction_id, factor_id, action_id = _DIRECTION_INDEX[direction], factor_index[factor], action_index[action]
        if not bool(candidate[direction_id, factor_id, action_id]):
            raise ValueError(f"IC-DOR edge {_edge_key(spec)} is not a candidate route")
        alternatives = torch.nonzero(candidate[direction_id, :, action_id], as_tuple=False).flatten()
        alternatives = alternatives[alternatives != factor_id]
        if alternatives.numel() == 0:
            raise ValueError(f"IC-DOR edge {_edge_key(spec)} has no equal-mass random candidate")
        random_factor = int(alternatives[0])
        on, off, random = base.clone(), base.clone(), base.clone()
        # Hold every other route fixed; within this action/direction the three arms have mass 1/0/1.
        on[direction_id, :, action_id] = False
        off[direction_id, :, action_id] = False
        random[direction_id, :, action_id] = False
        on[direction_id, factor_id, action_id] = True
        random[direction_id, random_factor, action_id] = True
        plans.append((spec, on, off, random, action_id))
    logits: dict[str, dict[str, list[torch.Tensor]]] = {
        _edge_key(spec): {"on": [], "off": [], "random": [], "labels": []} for spec, *_ in plans
    }
    kwargs = dict(forward_kwargs or {})
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise ValueError("IC-DOR audit loader must yield mapping batches")
            splits = _split_values(batch)
            images = _images(batch, device)
            labels = batch.get("action")
            if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape != (images.shape[0], len(action_names)):
                raise ValueError("IC-DOR edge audit requires [batch, action] action labels")
            if len(splits) != images.shape[0]:
                raise ValueError("IC-DOR audit split rows must align with images")
            for spec, on, off, random, action_id in plans:
                key = _edge_key(spec)
                for arm, mask in (("on", on), ("off", off), ("random", random)):
                    setter(mask.to(device=device, dtype=torch.bool))
                    output = _forward(model, images, "full", kwargs)
                    action_logits = output.get("action_final_logits", output.get("action_logits_raw"))
                    if not isinstance(action_logits, torch.Tensor) or action_logits.shape != labels.shape:
                        raise ValueError("IC-DOR edge audit forward requires action_final_logits [batch, action]")
                    logits[key][arm].append(action_logits[:, action_id].detach().float().cpu())
                logits[key]["labels"].append(labels[:, action_id].detach().float().cpu())
    finally:
        setter(base)
        model.train(was_training)
    if not any(values["labels"] for values in logits.values()):
        raise ValueError("IC-DOR edge audit loader produced no train_audit observations")
    generator = torch.Generator().manual_seed(bootstrap_seed)
    edge_stats: dict[str, dict[str, Any]] = {}
    for spec, *_ in plans:
        key = _edge_key(spec)
        values = {name: torch.cat(rows, dim=0) for name, rows in logits[key].items()}
        metrics = _edge_metrics(
            values["on"], values["off"], values["random"], values["labels"],
            direction=str(spec["direction"]),
        )
        replicate = {name: [] for name in metrics}
        for _ in range(bootstrap_replicates):
            indices = _stratified_indices(values["labels"] > 0.5, generator)
            sampled = _edge_metrics(
                values["on"][indices], values["off"][indices], values["random"][indices], values["labels"][indices],
                direction=str(spec["direction"]),
            )
            for name, value in sampled.items():
                replicate[name].append(value)
        intervals = {name: {"lower": _quantile(samples, 0.025), "upper": _quantile(samples, 0.975)} for name, samples in replicate.items()}
        edge_stats[key] = {
            "factor": str(spec["factor"]), "action": str(spec["action"]), "direction": str(spec["direction"]), "polarity": str(spec["polarity"]),
            "matched_counts": {"factor_on": int(values["on"].numel()), "factor_off": int(values["off"].numel()), "equal_mass_random": int(values["random"].numel())},
            "metrics": metrics,
            "bootstrap_ci95": intervals,
            "bootstrap_lcb95": {name: interval["lower"] for name, interval in intervals.items()},
        }
    return {"source_split": "train_audit", "edge_stats": edge_stats}
