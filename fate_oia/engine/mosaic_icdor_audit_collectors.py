from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from fate_oia.datasets.mosaic_icdor_factor_supervision import build_factor_supervision


_ABLATION_MODES = ("full", "content_only", "prior_only", "query_shuffled", "image_shuffled")
_DIRECTION_INDEX = {"support": 0, "veto": 1}


def build_matched_factor_controls(
    selected_mask: torch.Tensor,
    *,
    selected_factor_type: str,
    selected_region: str,
    identity_masks: torch.Tensor | None = None,
    identity_names: Sequence[str] = (),
    identity_types: Sequence[str] = (),
    identity_regions: Sequence[str] = (),
    selected_factor_index: int | None = None,
    region_mask: torch.Tensor | None = None,
    min_controls: int = 4,
) -> list[dict[str, Any]]:
    """Build independent equal-mass controls without reusing selected pixels."""
    if selected_mask.ndim != 3 or selected_mask.shape[0] != 1 or min_controls < 4:
        raise ValueError("IC-DOR matched controls require one [1,H,W] mask and min_controls >= 4")
    selected = selected_mask.bool()
    mass = int(selected.sum())
    if mass == 0:
        raise ValueError("IC-DOR matched controls require non-empty selected mass")
    if region_mask is None:
        eligible = torch.ones_like(selected, dtype=torch.bool)
    else:
        if region_mask.shape != selected.shape:
            raise ValueError("IC-DOR control region mask must align with selected mask")
        eligible = region_mask.bool()
    controls: list[dict[str, Any]] = []
    # Identical geometry may still represent distinct control provenance (for
    # example an identity-matched object versus a spatial roll), so dedupe only
    # within each arm type.
    seen: set[tuple[str, bytes]] = set()
    occupied = selected.clone()

    def add(mask: torch.Tensor, control_type: str, **provenance: Any) -> None:
        nonlocal occupied
        mask = mask.bool() & eligible & ~occupied
        if int(mask.sum()) != mass:
            return
        fingerprint = (control_type, mask.cpu().numpy().tobytes())
        if fingerprint in seen:
            return
        seen.add(fingerprint)
        occupied = occupied | mask
        controls.append({
            "schema_version": "icdor_matched_control.v1", "control_type": control_type,
            "mask": mask, "mask_sum": float(mask.sum()), "overlap": float((mask & selected).sum()),
            "mass_error": abs(float(mask.sum()) - mass) / mass,
            "factor_type": selected_factor_type, "region": selected_region,
            **provenance,
        })

    # Prefer genuine identity-matched controls before synthetic spatial arms so
    # spatial enumeration cannot consume every eligible support first.
    if identity_masks is not None:
        if identity_masks.ndim == 4 and identity_masks.shape[1] == 1:
            identity_masks = identity_masks[:, 0]
        if (
            identity_masks.ndim != 3
            or identity_masks.shape[1:] != selected.shape[1:]
            or len(identity_names) != identity_masks.shape[0]
            or len(identity_types) != identity_masks.shape[0]
            or len(identity_regions) != identity_masks.shape[0]
        ):
            raise ValueError("IC-DOR identity control metadata must align with [N,H,W] masks")
        for index, mask in enumerate(identity_masks):
            # An identity control must be a same-type *wrong* factor. Reusing
            # the selected factor would turn a spatial re-placement into a
            # falsely labelled identity intervention.
            if selected_factor_index is not None and index == selected_factor_index:
                continue
            if identity_types[index] == selected_factor_type and identity_regions[index] == selected_region:
                candidate_scores = mask.float().unsqueeze(0) * eligible.float() * (~occupied).float()
                available = candidate_scores.gt(0)
                if int(available.sum()) < mass:
                    continue
                flat_scores = candidate_scores.flatten()
                chosen = torch.topk(flat_scores, k=mass, largest=True, sorted=False).indices
                identity = torch.zeros_like(flat_scores, dtype=torch.bool)
                identity[chosen] = True
                add(
                    identity.reshape_as(selected), "same_type_identity",
                    identity_source_factor_index=index,
                    identity_source_factor_name=str(identity_names[index]),
                    identity_source_factor_type=str(identity_types[index]),
                    identity_source_region=str(identity_regions[index]),
                )
                if any(control["control_type"] == "same_type_identity" for control in controls):
                    break
    # Cyclic shifts preserve exact mass. Enumerating offsets lets irregular masks
    # reject overlaps rather than silently accepting a near-match.
    height, width = selected.shape[-2:]
    for vertical in range(height):
        for horizontal in range(width):
            if vertical or horizontal:
                add(
                    torch.roll(selected, shifts=(vertical, horizontal), dims=(-2, -1)),
                    "spatial_roll", spatial_offset=[vertical, horizontal],
                )
    if len(controls) < min_controls:
        raise ValueError("IC-DOR matched controls require at least four non-overlapping equal-mass arms")
    return controls


def _select_topk_continuous_evidence(
    source: torch.Tensor,
    *,
    eligible: torch.Tensor,
    evidence_slots: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Make an auditable sparse selected-evidence arm from a dense soft mask.

    The model's factor maps are sigmoid/continuous, so ``mask > 0`` is almost
    always the entire image. Controls must instead operate on the factor's own
    top-K evidence slots inside its declared semantic region. The selected arm
    retains the original continuous values at those slots; control arms later
    receive exactly the same value multiset at non-overlapping locations.
    """
    if source.ndim != 2 or eligible.shape != (1, *source.shape):
        raise ValueError("IC-DOR top-K evidence support must align with [H,W] source and [1,H,W] region")
    if evidence_slots <= 0:
        raise ValueError("IC-DOR evidence_slots must be positive")
    source = source.detach()
    eligible_2d = eligible.squeeze(0).bool()
    eligible_values = source.masked_fill(~eligible_2d, 0.0)
    positive = eligible_values > 0.0
    positive_count = int(positive.sum())
    source_region_mass = float(eligible_values.sum())
    sparse = torch.zeros_like(source)
    support = torch.zeros_like(source, dtype=torch.bool)
    if positive_count == 0 or source_region_mass <= 0.0:
        return sparse, support.unsqueeze(0), {
            "support_method": "topk_continuous_evidence",
            "selected_support_count": 0,
            "selected_mass": 0.0,
            "source_region_mass": source_region_mass,
            "selected_mass_fraction": 0.0,
        }
    slot_count = min(int(evidence_slots), positive_count)
    chosen = torch.topk(eligible_values.flatten(), k=slot_count, largest=True, sorted=False).indices
    support.flatten()[chosen] = True
    sparse[support] = source[support]
    selected_mass = float(sparse.sum())
    return sparse, support.unsqueeze(0), {
        "support_method": "topk_continuous_evidence",
        "selected_support_count": slot_count,
        "selected_mass": selected_mass,
        "source_region_mass": source_region_mass,
        "selected_mass_fraction": selected_mass / max(source_region_mass, 1e-12),
    }


def _place_continuous_values_on_control(
    selected_sparse: torch.Tensor,
    selected_support: torch.Tensor,
    control_mask: torch.Tensor,
) -> torch.Tensor:
    """Move, rather than rescale, selected continuous evidence onto a control."""
    if selected_support.shape != control_mask.shape or selected_support.shape != (1, *selected_sparse.shape):
        raise ValueError("IC-DOR control support must align with selected evidence")
    selected_values = selected_sparse[selected_support.squeeze(0)]
    control_positions = control_mask.squeeze(0).flatten().nonzero(as_tuple=False).flatten()
    if selected_values.numel() == 0 or selected_values.numel() != control_positions.numel():
        raise ValueError("IC-DOR matched control must preserve selected continuous slot count")
    replacement = torch.zeros_like(selected_sparse)
    replacement.flatten()[control_positions] = selected_values
    return replacement


def build_batch_matched_factor_selected_and_control_overrides(
    factor_masks: torch.Tensor,
    *,
    factor_index: int,
    factor: str,
    factor_type: str,
    region: str,
    identity_factor_types: Sequence[str] = (),
    identity_regions: Sequence[str] = (),
    identity_factor_names: Sequence[str] = (),
    arm_count: int = 4,
    evidence_slots: int = 16,
) -> tuple[torch.Tensor, list[torch.Tensor], list[list[dict[str, Any]]]]:
    """Build selected and matched-control interventions from batch-local masks.

    All five arms retain the same nonzero continuous evidence values. The
    selected arm is the factor's own top-K regional evidence; the first
    control is a different factor with the same type/region, followed by
    non-overlapping same-region spatial controls. No test label or geometry is
    consulted.
    """
    if factor_masks.ndim != 4 or not factor_masks.is_floating_point() or not 0 <= factor_index < factor_masks.shape[1]:
        raise ValueError("IC-DOR controls require floating [B,F,H,W] factor masks")
    if arm_count < 4:
        raise ValueError("IC-DOR controls require at least four arms")
    batch, _, height, width = factor_masks.shape
    region_mask = _semantic_region_mask(height, width, region=region, device=factor_masks.device)
    selected_override = factor_masks.detach().clone()
    overrides = [factor_masks.detach().clone() for _ in range(arm_count)]
    arm_records: list[list[dict[str, Any]]] = [[] for _ in range(arm_count)]
    for sample_index in range(batch):
        source = factor_masks[sample_index, factor_index].detach()
        selected_sparse, support, selected_stats = _select_topk_continuous_evidence(
            source, eligible=region_mask, evidence_slots=evidence_slots
        )
        selected_override[sample_index, factor_index] = selected_sparse

        def append_unavailable(reason: str) -> None:
            for arm_index in range(arm_count):
                arm_records[arm_index].append({
                    "schema_version": "icdor_matched_control.v3",
                    "arm_index": arm_index,
                    "sample_index": sample_index,
                    "factor": factor,
                    "selected_factor_index": factor_index,
                    "factor_type": factor_type,
                    "region": region,
                    "control_type": "unavailable_noop",
                    "available": False,
                    "unavailable_reason": reason,
                    "control_support_method": "topk_continuous_evidence",
                    "control_evidence_slots": int(evidence_slots),
                    "selected_mass": float(selected_stats["selected_mass"]),
                    "source_region_mass": float(selected_stats["source_region_mass"]),
                    "selected_mass_fraction": float(selected_stats["selected_mass_fraction"]),
                    "selected_support_count": int(selected_stats["selected_support_count"]),
                    "control_mass": 0.0,
                    "mass_error": 0.0,
                    "overlap": 0.0,
                })

        if not bool(support.any()):
            append_unavailable("empty_declared_region_topk_evidence")
            continue
        try:
            candidates = build_matched_factor_controls(
                support,
                selected_factor_type=factor_type,
                selected_region=region,
                identity_masks=factor_masks[sample_index],
                identity_names=identity_factor_names,
                identity_types=identity_factor_types,
                identity_regions=identity_regions,
                selected_factor_index=factor_index,
                region_mask=region_mask,
                min_controls=arm_count,
            )
            identity_controls = [item for item in candidates if item["control_type"] == "same_type_identity"]
            spatial_controls = [item for item in candidates if item["control_type"] == "spatial_roll"]
            if not identity_controls or len(spatial_controls) < arm_count - 1:
                raise ValueError("IC-DOR matched controls require identity and spatial arms")
            controls = [identity_controls[0], *spatial_controls[: arm_count - 1]]
        except ValueError as error:
            if not any(
                marker in str(error)
                for marker in (
                    "at least four non-overlapping equal-mass arms",
                    "require identity and spatial arms",
                )
            ):
                raise
            append_unavailable("insufficient_identity_or_spatial_equal_mass_controls")
            continue
        selected_mass = float(selected_sparse.sum())
        for arm_index, control in enumerate(controls):
            replacement = _place_continuous_values_on_control(selected_sparse, support, control["mask"])
            overrides[arm_index][sample_index, factor_index] = replacement
            control_mass = float(replacement.sum())
            mass_error = abs(control_mass - selected_mass) / max(selected_mass, 1e-12)
            if mass_error > 0.05 or float(control["overlap"]) != 0.0:
                raise ValueError("IC-DOR matched control violates non-overlap or mass tolerance")
            arm_records[arm_index].append({
                "schema_version": "icdor_matched_control.v3",
                "arm_index": arm_index,
                "sample_index": sample_index,
                "factor": factor,
                "selected_factor_index": factor_index,
                "factor_type": factor_type,
                "region": region,
                "control_type": control["control_type"],
                "available": True,
                "control_support_method": "topk_continuous_evidence",
                "control_evidence_slots": int(evidence_slots),
                "selected_mass": selected_mass,
                "source_region_mass": float(selected_stats["source_region_mass"]),
                "selected_mass_fraction": float(selected_stats["selected_mass_fraction"]),
                "selected_support_count": int(selected_stats["selected_support_count"]),
                "control_mass": control_mass,
                "mass_error": mass_error,
                "overlap": 0.0,
                **{
                    key: control[key]
                    for key in (
                        "identity_source_factor_index", "identity_source_factor_name",
                        "identity_source_factor_type", "identity_source_region", "spatial_offset",
                    )
                    if key in control
                },
            })
    return selected_override, overrides, arm_records


def factor_control_spec(model: torch.nn.Module, *, factor_name: str, factor_index: int) -> dict[str, str]:
    """Return image-only semantic provenance for a selected factor."""
    ontology = getattr(model, "ontology", None)
    factors = ontology.get("factors") if isinstance(ontology, Mapping) else None
    if isinstance(factors, Sequence) and factor_index < len(factors) and isinstance(factors[factor_index], Mapping):
        factor = factors[factor_index]
        if str(factor.get("name", factor_name)) == factor_name:
            declared_regions = factor.get("weak_regions")
            if isinstance(declared_regions, Sequence) and not isinstance(declared_regions, (str, bytes)):
                declared_region = str(declared_regions[0]) if len(declared_regions) == 1 else "unspecified"
            else:
                declared_region = "unspecified"
            return {
                "factor": factor_name,
                "factor_type": str(factor.get("type", "unknown")),
                "region": str(factor.get("spatial", factor.get("region_prior", declared_region))),
            }
    return {"factor": factor_name, "factor_type": "unknown", "region": "unspecified"}


def _semantic_region_mask(height: int, width: int, *, region: str, device: torch.device) -> torch.Tensor:
    specifications = {
        "upper_front": (0.0, -0.65, 0.55, 0.32),
        "front_center": (0.0, 0.25, 0.45, 0.55),
        "left_corridor": (-0.52, 0.38, 0.38, 0.62),
        "right_corridor": (0.52, 0.38, 0.38, 0.62),
        "center_corridor": (0.0, 0.52, 0.38, 0.58),
    }
    if region not in specifications:
        return torch.ones(1, height, width, dtype=torch.bool, device=device)
    center_x, center_y, scale_x, scale_y = specifications[region]
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    return ((((x - center_x) / scale_x) ** 2 + ((y - center_y) / scale_y) ** 2) <= 1.0).unsqueeze(0)


def build_batch_matched_factor_control_overrides(
    factor_masks: torch.Tensor,
    *,
    factor_index: int,
    factor: str,
    factor_type: str,
    region: str,
    identity_factor_types: Sequence[str] = (),
    identity_regions: Sequence[str] = (),
    identity_factor_names: Sequence[str] = (),
    arm_count: int = 4,
) -> tuple[list[torch.Tensor], list[list[dict[str, Any]]]]:
    """Backward-compatible control-only view of the selected/control builder."""
    _, overrides, arm_records = build_batch_matched_factor_selected_and_control_overrides(
        factor_masks,
        factor_index=factor_index,
        factor=factor,
        factor_type=factor_type,
        region=region,
        identity_factor_types=identity_factor_types,
        identity_regions=identity_regions,
        identity_factor_names=identity_factor_names,
        arm_count=arm_count,
    )
    return overrides, arm_records


def summarize_matched_control_arms(arm_records: Sequence[Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Persist per-arm provenance without storing a duplicate tensor for every forward."""
    summaries: list[dict[str, Any]] = []
    for records in arm_records:
        if not records:
            raise ValueError("IC-DOR matched control arm has no provenance rows")
        available = [record for record in records if bool(record.get("available", True))]
        if not available:
            first = records[0]
            summaries.append({
                "schema_version": "icdor_matched_control.v3",
                "arm_index": int(first["arm_index"]),
                "factor": str(first["factor"]),
                "factor_type": str(first["factor_type"]),
                "region": str(first["region"]),
                "control_type": "unavailable_noop",
                "sample_count": len(records),
                "available_sample_count": 0,
                "max_mass_error": None,
                "max_overlap": None,
                "selected_mass_total": 0.0,
                "control_mass_total": 0.0,
                "control_support_method": str(first.get("control_support_method", "topk_continuous_evidence")),
                "control_evidence_slots": int(first.get("control_evidence_slots", 0)),
                "selected_support_count_mean": 0.0,
                "selected_mass_fraction_mean": 0.0,
                "source_region_mass_total": 0.0,
                "selected_factor_indices": sorted({
                    int(record["selected_factor_index"])
                    for record in records if "selected_factor_index" in record
                }),
                "unavailable_reason": str(first.get("unavailable_reason", "insufficient_matched_controls")),
                "identity_source_factor_indices": [],
                "identity_source_factor_names": [],
                "identity_source_factor_types": [],
                "identity_source_regions": [],
                "spatial_offsets": [],
            })
            continue
        first = available[0]
        summaries.append({
            "schema_version": "icdor_matched_control.v3",
            "arm_index": int(first["arm_index"]),
            "factor": str(first["factor"]),
            "factor_type": str(first["factor_type"]),
            "region": str(first["region"]),
            "control_type": str(first["control_type"]),
            "sample_count": len(records),
            "available_sample_count": sum(bool(record.get("available", True)) for record in records),
            "max_mass_error": max(float(record["mass_error"]) for record in available),
            "max_overlap": max(float(record["overlap"]) for record in available),
            "selected_mass_total": sum(float(record["selected_mass"]) for record in available),
            "control_mass_total": sum(float(record["control_mass"]) for record in available),
            "control_support_method": str(first.get("control_support_method", "topk_continuous_evidence")),
            "control_evidence_slots": int(first.get("control_evidence_slots", 0)),
            "selected_support_count_mean": sum(
                int(record.get("selected_support_count", 0)) for record in available
            ) / len(available),
            "selected_mass_fraction_mean": sum(
                float(record.get("selected_mass_fraction", 0.0)) for record in available
            ) / len(available),
            "source_region_mass_total": sum(
                float(record.get("source_region_mass", 0.0)) for record in available
            ),
            "selected_factor_indices": sorted({
                int(record["selected_factor_index"])
                for record in available if "selected_factor_index" in record
            }),
            "identity_source_factor_indices": sorted({
                int(record["identity_source_factor_index"])
                for record in available if "identity_source_factor_index" in record
            }),
            "identity_source_factor_names": sorted({
                str(record["identity_source_factor_name"])
                for record in available if "identity_source_factor_name" in record
            }),
            "identity_source_factor_types": sorted({
                str(record["identity_source_factor_type"])
                for record in available if "identity_source_factor_type" in record
            }),
            "identity_source_regions": sorted({
                str(record["identity_source_region"])
                for record in available if "identity_source_region" in record
            }),
            "spatial_offsets": sorted({
                tuple(int(value) for value in record["spatial_offset"])
                for record in available if "spatial_offset" in record
            }),
        })
    return summaries


def _split_values(batch: Mapping[str, Any], *, expected_split: str = "train_audit") -> list[str]:
    split = batch.get("split")
    values = [split] if isinstance(split, str) else list(split) if isinstance(split, Sequence) else []
    if not values or any(value != expected_split for value in values):
        raise ValueError(f"IC-DOR audit collectors accept {expected_split} batches only")
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


def summarize_factor_supervision(
    scores: torch.Tensor,
    confirmed_positive: torch.Tensor,
    reliable_negative: torch.Tensor,
    weak_negative: torch.Tensor,
    *,
    geometry_valid_mask: torch.Tensor,
) -> dict[str, Any]:
    """Summarize the strongest honest audit supported by available labels."""
    tensors = (scores, confirmed_positive, reliable_negative, weak_negative, geometry_valid_mask)
    if any(value.ndim != 1 or value.shape != scores.shape for value in tensors):
        raise ValueError("factor supervision summary expects aligned 1-D tensors")
    if any(value.dtype != torch.bool for value in tensors[1:]):
        raise ValueError("factor supervision masks must be boolean")
    if bool((confirmed_positive & (reliable_negative | weak_negative)).any()):
        raise ValueError("factor supervision positive and negative masks must be disjoint")

    if bool(confirmed_positive.any()) and bool(reliable_negative.any()):
        mode, negative = "binary_confirmed", reliable_negative
        ceiling = "Certified" if bool((geometry_valid_mask & confirmed_positive).any()) else "Reason-only"
    elif bool(confirmed_positive.any()) and bool(weak_negative.any()):
        mode, negative = "positive_vs_weak_negative", weak_negative
        enough_geometry = int((geometry_valid_mask & confirmed_positive).sum()) >= 2
        ceiling = "Certified" if enough_geometry else "Reason-only"
    elif bool(confirmed_positive.any()):
        return {
            "evaluation_mode": "positive_only", "metric_available": False,
            "presence_auprc": None, "unavailable_reason": "no_negative_observations",
            "certificate_ceiling": "Abstained",
        }
    else:
        return {
            "evaluation_mode": "unavailable", "metric_available": False,
            "presence_auprc": None, "unavailable_reason": "no_confirmed_positive_observations",
            "certificate_ceiling": "Abstained",
        }

    rows = confirmed_positive | negative
    auprc = _average_precision(scores[rows], confirmed_positive[rows])
    return {
        "evaluation_mode": mode, "metric_available": True,
        "presence_auprc": auprc, "unavailable_reason": None,
        "certificate_ceiling": ceiling,
    }


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


def _grounding_delta(mask: torch.Tensor, geometry: torch.Tensor, valid: torch.Tensor, generator: torch.Generator) -> float | None:
    rows = torch.nonzero(valid.bool(), as_tuple=False).flatten()
    if rows.numel() < 2:
        # A matched random geometry is undefined with fewer than two rows.
        return None
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
    factor_definitions: Sequence[Mapping[str, Any]] | None = None,
    device: torch.device,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 0,
    forward_kwargs: Mapping[str, Any] | None = None,
    source_split: str = "train_audit",
) -> dict[str, Any]:
    """Collect certificate-ready IC-DOR factor statistics from train_audit only.

    ``audit_views`` must contain at least two BDD100K views registered to the base
    image coordinate system.  The collector never synthesizes views or labels.
    """
    if not factor_names or len(set(factor_names)) != len(factor_names):
        raise ValueError("IC-DOR factor audit requires unique factor names")
    if bootstrap_replicates < 1:
        raise ValueError("IC-DOR factor audit requires positive bootstrap_replicates")
    if source_split not in {"train_audit", "audit_visual"}:
        raise ValueError("IC-DOR factor audit source split is invalid")
    if source_split == "audit_visual" and factor_definitions is not None:
        raise ValueError("audit_visual credibility must not use reason-anchor factor supervision")
    kwargs = dict(forward_kwargs or {})
    factor_count = len(factor_names)
    collected: dict[str, list[torch.Tensor]] = {
        "target": [], "known": [], "weak": [], "geometry_known": [], "geometry": [],
        "full": [], "content_only": [], "prior_only": [], "query_shuffled": [], "image_shuffled": [],
        "full_mask": [], "view_mask": [], "mirror_mask": [], "visibility": [], "prototype_weights": [],
        "view_consistency": [], "mirror_consistency": [],
    }
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            if not isinstance(batch, Mapping):
                raise ValueError("IC-DOR audit loader must yield mapping batches")
            splits = _split_values(batch, expected_split=source_split)
            images = _images(batch, device)
            if len(splits) != images.shape[0]:
                raise ValueError("IC-DOR audit split rows must align with images")
            grounding = grounding_builder(_records(batch, images.shape[0]), device=device, split="train")
            if not isinstance(grounding, Mapping):
                raise ValueError("IC-DOR grounding builder must return a mapping")
            target = _tensor(grounding, "presence_target", rows=images.shape[0], columns=factor_count)
            known = _tensor(grounding, "presence_known_mask", rows=images.shape[0], columns=factor_count).bool()
            weak = _tensor(grounding, "weak_negative_mask", rows=images.shape[0], columns=factor_count).bool()
            if factor_definitions is not None:
                if len(factor_definitions) != factor_count:
                    raise ValueError("IC-DOR factor definitions must align with factor names")
                reasons = batch.get("reason")
                if not isinstance(reasons, torch.Tensor) or reasons.shape[0] != images.shape[0]:
                    raise ValueError("IC-DOR reason-anchor audit requires aligned reason targets")
                supervision = build_factor_supervision(
                    grounding,
                    reasons.to(device),
                    factor_definitions,
                    split="train_audit",
                )
                target = supervision["supervision_target"].detach().float().cpu()
                positive = (supervision["geometry_positive_mask"] | supervision["positive_anchor_mask"]).detach().cpu()
                reliable_negative = supervision["reliable_negative_mask"].detach().cpu()
                known = positive | reliable_negative
                weak = supervision["weak_negative_mask"].detach().cpu()
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
            full_mask = collected["full_mask"][-1]
            view_mask = collected["view_mask"][-1]
            mirror_mask = collected["mirror_mask"][-1]
            collected["view_consistency"].append((1.0 - (full_mask - view_mask).abs().mean(dim=(-2, -1))).clamp(0.0, 1.0))
            collected["mirror_consistency"].append((1.0 - (full_mask - mirror_mask).abs().mean(dim=(-2, -1))).clamp(0.0, 1.0))
            collected["visibility"].append(_tensor(full, "factor_visibility_prob", rows=images.shape[0], columns=factor_count))
            collected["prototype_weights"].append(_tensor(full, "prototype_weights", rows=images.shape[0], columns=factor_count))
    finally:
        model.train(was_training)
    if not collected["target"]:
        raise ValueError("IC-DOR factor audit loader produced no train_audit observations")
    values = {key: torch.cat(items, dim=0) for key, items in collected.items()}
    generator = torch.Generator().manual_seed(bootstrap_seed)
    factor_stats: dict[str, dict[str, Any]] = {}
    unknown_rows_total = 0
    unknown_rows_in_metric_total = 0
    for column, name in enumerate(factor_names):
        target, known, weak = values["target"][:, column], values["known"][:, column], values["weak"][:, column]
        reliable_negative = known & (target <= 0.5)
        confirmed_positive = known & (target > 0.5)
        full = values["full"][:, column]
        prior = values["prior_only"][:, column]
        content = values["content_only"][:, column]
        query = values["query_shuffled"][:, column]
        image = values["image_shuffled"][:, column]
        geometry_valid = values["geometry_known"][:, column]
        summary = summarize_factor_supervision(
            full, confirmed_positive, reliable_negative, weak,
            geometry_valid_mask=geometry_valid,
        )
        if summary["metric_available"]:
            negative = reliable_negative if summary["evaluation_mode"] == "binary_confirmed" else weak
            metric_rows = confirmed_positive | negative
            target_metric = confirmed_positive[metric_rows]
            score_full = _average_precision(full[metric_rows], target_metric)
            score_prior = _average_precision(prior[metric_rows], target_metric)
            score_content = _average_precision(content[metric_rows], target_metric)
            score_query = _average_precision(query[metric_rows], target_metric)
            score_image = _average_precision(image[metric_rows], target_metric)
        else:
            metric_rows = torch.zeros_like(confirmed_positive)
            target_metric = confirmed_positive[metric_rows]
            score_full = score_prior = score_content = score_query = score_image = None
        unknown_rows = ~known & ~weak
        unknown_rows_total += int(unknown_rows.sum())
        unknown_rows_in_metric_total += int((unknown_rows & metric_rows).sum())
        grounding_delta = _grounding_delta(values["full_mask"][:, column], values["geometry"][:, column], geometry_valid, generator)
        mask_difference = (values["full_mask"][:, column] - values["view_mask"][:, column]).abs()
        view_consistency = float((1.0 - mask_difference.mean()).clamp(0.0, 1.0))
        mirror_difference = (values["full_mask"][:, column] - values["mirror_mask"][:, column]).abs()
        mirror_consistency = float((1.0 - mirror_difference.mean()).clamp(0.0, 1.0))
        effective, dominant, dead = _prototype_stats(values["prototype_weights"][:, column])
        replicate: dict[str, list[float]] = {
            "full_minus_prior_only": [], "query_shuffle_drop": [], "image_shuffle_drop": [], "grounding_minus_random": [],
            "stability": [],
        }
        metric_indices = torch.nonzero(metric_rows, as_tuple=False).flatten()
        for _ in range(bootstrap_replicates):
            if summary["metric_available"]:
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
                stability = torch.minimum(values["view_consistency"][sample, column], values["mirror_consistency"][sample, column])
                replicate["stability"].append(float((stability > 0.75).float().mean()))
            geometry_sample = torch.nonzero(geometry_valid, as_tuple=False).flatten()
            if geometry_sample.numel() >= 2:
                draw = geometry_sample.index_select(0, torch.randint(geometry_sample.numel(), (geometry_sample.numel(),), generator=generator))
                value = _grounding_delta(values["full_mask"][draw, column], values["geometry"][draw, column], torch.ones(draw.numel(), dtype=torch.bool), generator)
                if value is not None:
                    replicate["grounding_minus_random"].append(value)
        factor_stats[str(name)] = {
            **summary,
            "counts": {
                "confirmed_positive": int(confirmed_positive.sum()),
                "reliable_negative": int(reliable_negative.sum()),
                "weak_negative": int((weak & ~known).sum()),
                "unknown": int(unknown_rows.sum()),
                "geometry_valid": int(geometry_valid.sum()),
            },
            "scores": {
                "full": score_full,
                "content_only": score_content,
                "prior_only": score_prior,
                "query_shuffle_drop": None if score_full is None else score_full - score_query,
                "image_shuffle_drop": None if score_full is None else score_full - score_image,
                "grounding_minus_random": grounding_delta,
                "view_consistency": view_consistency,
                "mirror_consistency": mirror_consistency,
                "ece": None if not summary["metric_available"] else _expected_calibration_error(full[metric_rows], target_metric.float()),
                "presence_variance": None if not summary["metric_available"] else float(full[metric_rows].var(unbiased=False)),
                "visibility_variance": float(values["visibility"][:, column].var(unbiased=False)),
            },
            "prototype": {"effective_count": effective, "dominant_rate": dominant, "dead_count": dead},
            "bootstrap_lcb95": {key: (_quantile(samples, 0.05) if samples else None) for key, samples in replicate.items()},
            "bootstrap_positive_rate": {
                key: (float(torch.tensor(samples, dtype=torch.float32).gt(0.0).float().mean()) if samples else 0.0)
                for key, samples in replicate.items()
            },
        }
    if unknown_rows_in_metric_total != 0:
        raise RuntimeError("IC-DOR factor audit leaked unknown rows into binary metrics")
    return {
        "source_split": source_split,
        "row_count": int(values["target"].shape[0]),
        "factor_count": len(factor_stats),
        "factor_stats": factor_stats,
        "audit_integrity": {
            "collector_completed": True,
            "exception": None,
            "unknown_policy": "excluded_from_binary_metrics",
            "unknown_rows_total": unknown_rows_total,
            "unknown_rows_in_metric_total": unknown_rows_in_metric_total,
        },
    }


def _edge_metrics(
    on: torch.Tensor,
    off: torch.Tensor,
    random: torch.Tensor,
    labels: torch.Tensor,
    *,
    direction: str,
    random_identity: torch.Tensor,
    random_spatial: torch.Tensor,
) -> dict[str, float] | None:
    probability_on, probability_off, probability_random = on.sigmoid(), off.sigmoid(), random.sigmoid()
    probability_identity = random_identity.sigmoid()
    probability_spatial = random_spatial.sigmoid()
    positive = labels > 0.5
    if not bool(positive.any()) or bool(positive.all()):
        # A bounded smoke may legitimately contain only one class for an
        # action.  It cannot estimate a target-effect metric, but it must not
        # fabricate a zero effect or abort all independent diagnostics.
        return None
    if direction == "support":
        evaluation_rows = positive
        signed = probability_on - probability_off
        random_signed = probability_random - probability_off
        identity_signed = probability_identity - probability_off
        spatial_signed = probability_spatial - probability_off
    elif direction == "veto":
        evaluation_rows = ~positive
        signed = probability_off - probability_on
        random_signed = probability_off - probability_random
        identity_signed = probability_off - probability_identity
        spatial_signed = probability_off - probability_spatial
    else:
        raise ValueError("IC-DOR edge direction must be support or veto")
    if not bool(evaluation_rows.any()):
        return None
    tet = float(signed[evaluation_rows].mean())
    random_effect = float(random_signed[evaluation_rows].mean())
    identity_effect = float(identity_signed[evaluation_rows].mean())
    spatial_effect = float(spatial_signed[evaluation_rows].mean())
    calibration_on = 1.0 - (probability_on - labels).abs()
    calibration_off = 1.0 - (probability_off - labels).abs()
    isolated_edge_ap = _average_precision(probability_on, positive)
    visual_ap = _average_precision(probability_off, positive)
    return {
        "signed_effect": tet,
        "tet": tet,
        "tes": tet - random_effect,
        "tes_identity": tet - identity_effect,
        "tes_spatial": tet - spatial_effect,
        "cca": float((signed[evaluation_rows] > 0.0).float().mean()),
        "calibration_gain": float((calibration_on[evaluation_rows] - calibration_off[evaluation_rows]).mean()),
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
    source_split: str = "train_audit",
) -> dict[str, Any]:
    """Run matched on/off/equal-mass-random edge interventions on audit targets.

    ``audit_target`` is disjoint from ``audit_visual``: it may use target
    labels to measure utility but cannot update visual credibility.
    ``train_audit`` remains readable solely for historical v3 artifacts.
    """
    if not edge_specs or bootstrap_replicates < 1:
        raise ValueError("IC-DOR edge audit requires edges and positive bootstrap_replicates")
    if source_split not in {"audit_target", "train_audit"}:
        raise ValueError("IC-DOR edge audit source must be audit_target")
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
    plans: list[tuple[Mapping[str, Any], torch.Tensor, torch.Tensor, int, int]] = []
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
        on, off = base.clone(), base.clone()
        # Hold every other route fixed; random controls replace the selected
        # factor's spatial mask rather than selecting a different factor.
        on[direction_id, :, action_id] = False
        off[direction_id, :, action_id] = False
        on[direction_id, factor_id, action_id] = True
        plans.append((spec, on, off, action_id, factor_id))
    logits: dict[str, dict[str, list[torch.Tensor]]] = {
        _edge_key(spec): {
            "on": [], "off": [], "random": [], "random_identity": [],
            "random_spatial": [], "labels": [],
        } for spec, *_ in plans
    }
    kwargs = dict(forward_kwargs or {})
    if "factor_mask_override" in kwargs:
        raise ValueError("IC-DOR edge audit owns factor_mask_override")
    control_records: dict[str, list[list[dict[str, Any]]]] = {
        _edge_key(spec): [[] for _ in range(4)] for spec, *_ in plans
    }
    identity_specs = [
        factor_control_spec(model, factor_name=str(name), factor_index=index)
        for index, name in enumerate(factor_names)
    ]
    identity_factor_types = tuple(spec["factor_type"] for spec in identity_specs)
    identity_regions = tuple(spec["region"] for spec in identity_specs)
    identity_factor_names = tuple(spec["factor"] for spec in identity_specs)
    saw_batch = False
    was_training = model.training
    model.eval()
    try:
        for batch in loader:
            saw_batch = True
            if not isinstance(batch, Mapping):
                raise ValueError("IC-DOR audit loader must yield mapping batches")
            splits = _split_values(batch, expected_split=source_split)
            images = _images(batch, device)
            labels = batch.get("action")
            if not isinstance(labels, torch.Tensor) or labels.ndim != 2 or labels.shape != (images.shape[0], len(action_names)):
                raise ValueError("IC-DOR edge audit requires [batch, action] action labels")
            if len(splits) != images.shape[0]:
                raise ValueError("IC-DOR audit split rows must align with images")
            for spec, on, off, action_id, factor_id in plans:
                key = _edge_key(spec)
                setter(on.to(device=device, dtype=torch.bool))
                on_output = _forward(model, images, "full", {**kwargs, "return_masks": True})
                factor_masks = on_output.get("factor_soft_masks")
                if not isinstance(factor_masks, torch.Tensor) or factor_masks.shape[:2] != (images.shape[0], len(factor_names)):
                    raise ValueError("IC-DOR edge audit requires real factor_soft_masks [batch, factor, height, width]")
                if factor_masks.ndim != 4:
                    raise ValueError("IC-DOR edge audit factor_soft_masks must be [batch, factor, height, width]")
                factor_spec = factor_control_spec(model, factor_name=str(spec["factor"]), factor_index=factor_id)
                selected_override, overrides, arm_rows = build_batch_matched_factor_selected_and_control_overrides(
                    factor_masks,
                    factor_index=factor_id,
                    factor=factor_spec["factor"],
                    factor_type=factor_spec["factor_type"],
                    region=factor_spec["region"],
                    identity_factor_types=identity_factor_types,
                    identity_regions=identity_regions,
                    identity_factor_names=identity_factor_names,
                )
                for arm_index, rows in enumerate(arm_rows):
                    control_records[key][arm_index].extend(rows)
                common_available = torch.tensor(
                    [
                        all(bool(arm_rows[arm][sample].get("available")) for arm in range(4))
                        for sample in range(images.shape[0])
                    ],
                    dtype=torch.bool,
                )
                arm_outputs = []
                selected_output = _forward(
                    model, images, "full", {**kwargs, "factor_mask_override": selected_override}
                )
                for override in overrides:
                    output = _forward(model, images, "full", {**kwargs, "factor_mask_override": override})
                    action_logits = output.get("action_final_logits", output.get("action_logits_raw"))
                    if not isinstance(action_logits, torch.Tensor) or action_logits.shape != labels.shape:
                        raise ValueError("IC-DOR edge audit forward requires action_final_logits [batch, action]")
                    arm_outputs.append(action_logits[:, action_id].detach().float().cpu()[common_available])
                # Compare selected top-K factor evidence against controls with
                # the same continuous mass, never against the original dense
                # sigmoid map.
                action_logits = selected_output.get("action_final_logits", selected_output.get("action_logits_raw"))
                if not isinstance(action_logits, torch.Tensor) or action_logits.shape != labels.shape:
                    raise ValueError("IC-DOR edge audit forward requires action_final_logits [batch, action]")
                if not bool(common_available.any()):
                    continue
                logits[key]["on"].append(action_logits[:, action_id].detach().float().cpu()[common_available])
                setter(off.to(device=device, dtype=torch.bool))
                output = _forward(model, images, "full", kwargs)
                action_logits = output.get("action_final_logits", output.get("action_logits_raw"))
                if not isinstance(action_logits, torch.Tensor) or action_logits.shape != labels.shape:
                    raise ValueError("IC-DOR edge audit forward requires action_final_logits [batch, action]")
                logits[key]["off"].append(action_logits[:, action_id].detach().float().cpu()[common_available])
                logits[key]["random"].append(torch.stack(arm_outputs, dim=0).mean(dim=0))
                logits[key]["random_identity"].append(arm_outputs[0])
                logits[key]["random_spatial"].append(torch.stack(arm_outputs[1:], dim=0).mean(dim=0))
                logits[key]["labels"].append(labels[:, action_id].detach().float().cpu()[common_available])
    finally:
        setter(base)
        model.train(was_training)
    if not saw_batch:
        raise ValueError(f"IC-DOR edge audit loader produced no {source_split} observations")
    generator = torch.Generator().manual_seed(bootstrap_seed)
    edge_stats: dict[str, dict[str, Any]] = {}
    for spec, *_ in plans:
        key = _edge_key(spec)
        if not logits[key]["labels"]:
            edge_stats[key] = {
                "factor": str(spec["factor"]), "action": str(spec["action"]),
                "direction": str(spec["direction"]), "polarity": str(spec["polarity"]),
                "available": False,
                "unavailable_reason": "insufficient_matched_control_rows",
                "matched_counts": {
                    "factor_on": 0, "factor_off": 0, "equal_mass_random": 0,
                    "same_type_identity": 0, "spatial_roll": 0,
                },
                "matched_control_arms": summarize_matched_control_arms(control_records[key]),
                "metrics": None, "bootstrap_ci95": None, "bootstrap_lcb95": None,
            }
            continue
        values = {name: torch.cat(rows, dim=0) for name, rows in logits[key].items()}
        metrics = _edge_metrics(
            values["on"], values["off"], values["random"], values["labels"],
            direction=str(spec["direction"]),
            random_identity=values["random_identity"],
            random_spatial=values["random_spatial"],
        )
        if metrics is None:
            edge_stats[key] = {
                "factor": str(spec["factor"]), "action": str(spec["action"]),
                "direction": str(spec["direction"]), "polarity": str(spec["polarity"]),
                "available": False,
                "unavailable_reason": "insufficient_action_class_coverage",
                "matched_counts": {
                    "factor_on": int(values["on"].numel()),
                    "factor_off": int(values["off"].numel()),
                    "equal_mass_random": int(values["random"].numel()),
                    "same_type_identity": int(values["random_identity"].numel()),
                    "spatial_roll": int(values["random_spatial"].numel()),
                },
                "matched_control_arms": summarize_matched_control_arms(control_records[key]),
                "metrics": None, "bootstrap_ci95": None, "bootstrap_lcb95": None,
            }
            continue
        # Calibration gain is reported as a separate descriptive metric; it is
        # not an admission statistic and therefore has no bootstrap gate.
        replicate = {name: [] for name in metrics if name != "calibration_gain"}
        for _ in range(bootstrap_replicates):
            indices = _stratified_indices(values["labels"] > 0.5, generator)
            sampled = _edge_metrics(
                values["on"][indices], values["off"][indices], values["random"][indices], values["labels"][indices],
                direction=str(spec["direction"]),
                random_identity=values["random_identity"][indices],
                random_spatial=values["random_spatial"][indices],
            )
            if sampled is None:
                raise RuntimeError("IC-DOR stratified edge bootstrap lost required class coverage")
            for name in replicate:
                replicate[name].append(sampled[name])
        intervals = {name: {"lower": _quantile(samples, 0.025), "upper": _quantile(samples, 0.975)} for name, samples in replicate.items()}
        edge_stats[key] = {
            "factor": str(spec["factor"]), "action": str(spec["action"]), "direction": str(spec["direction"]), "polarity": str(spec["polarity"]),
            "matched_counts": {
                "factor_on": int(values["on"].numel()),
                "factor_off": int(values["off"].numel()),
                "equal_mass_random": int(values["random"].numel()),
                "same_type_identity": int(values["random_identity"].numel()),
                "spatial_roll": int(values["random_spatial"].numel()),
            },
            "matched_control_arms": summarize_matched_control_arms(control_records[key]),
            "available": True,
            "metrics": metrics,
            "bootstrap_ci95": intervals,
            "bootstrap_lcb95": {name: interval["lower"] for name, interval in intervals.items()},
        }
    return {"source_split": source_split, "edge_stats": edge_stats}
