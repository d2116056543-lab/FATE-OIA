"""P14 analytical deletion and public-field counterfactual objectives.

This module intentionally accepts only an already encoded shared field and
public readout/contribution callbacks.  It owns no raw-input, encoder,
persistence, loop-manager, or evaluation interface.  Discrete selection is
detached; gradients from the counterfactual loss flow only through the supplied
public callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import numbers
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from fate_oia.losses.rael_pu_losses import canonicalize_sample_id


PUBLIC_SLOT_COUNT = 20
COUNTERFACTUAL_EVERY_UPDATES = 8
VERTICAL_TOLERANCE = 0.10
MASS_TOLERANCE = 0.05
MAX_OVERLAP = 0.05
MASK_THRESHOLD = 0.50


def _require_finite(name: str, value: Tensor) -> None:
    """Validate P14's infrequent public inputs before any discrete routing."""

    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def _assert_finite_async(name: str, value: Tensor) -> None:
    """Fail closed through an async device assertion without host extraction."""

    torch._assert_async(torch.isfinite(value.detach()).all(), f"{name} must contain only finite values")


def _all_finite(*values: Tensor) -> Tensor:
    """Return one device-resident finite flag for the final availability sync."""

    return torch.stack([torch.isfinite(value.detach()).all() for value in values]).all()


def _require_positive_update(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a non-bool positive integer")
    return int(value)


def _canonical_case_ids(case_ids: Sequence[str | int], *, batch: int) -> list[str]:
    """Reuse P12's strict, lexical BDD image identity contract."""

    if isinstance(case_ids, (str, bytes, dict, Tensor)):
        raise TypeError("case_ids must be a sequence of non-bool integers or ASCII Windows image paths")
    try:
        values = list(case_ids)
    except TypeError as error:
        raise TypeError("case_ids must be a sequence of non-bool integers or ASCII Windows image paths") from error
    if len(values) != batch:
        raise ValueError("case_ids must provide one identifier per batch row")
    return [canonicalize_sample_id(value) for value in values]


def _diagnostics(
    *,
    device: torch.device,
    dtype: torch.dtype,
    optimizer_update: int,
    available: bool,
    computed: bool,
    margin: float,
    values: dict[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Build a fixed-device, fixed-shape diagnostic schema without fake zeros."""

    values = values or {}
    unset = torch.full((), float("nan"), device=device, dtype=dtype)

    def scalar(name: str) -> Tensor:
        value = values.get(name)
        if value is None:
            return unset.clone()
        if value.numel() != 1:
            raise ValueError(f"diagnostic {name} must be scalar")
        return value.detach().to(device=device, dtype=dtype).reshape(())

    return {
        "available": torch.tensor(available, device=device, dtype=torch.bool),
        "computed": torch.tensor(computed, device=device, dtype=torch.bool),
        "optimizer_update": torch.tensor(int(optimizer_update), device=device, dtype=torch.int64),
        "selected_effect": scalar("selected_effect"),
        "control_effect": scalar("control_effect"),
        "target_effect": scalar("target_effect"),
        "wrong_effect": scalar("wrong_effect"),
        "positive_analytical_effect": scalar("positive_analytical_effect"),
        "negative_analytical_effect": scalar("negative_analytical_effect"),
        "margin": torch.tensor(float(margin), device=device, dtype=dtype),
    }


def _require_public_contributions(unary_postgamma: Tensor, incident_pair_postgamma: Tensor) -> None:
    if unary_postgamma.ndim != 3 or incident_pair_postgamma.ndim != 3:
        raise ValueError("unary_postgamma and incident_pair_postgamma must be [B,K,20 public slots]")
    if unary_postgamma.shape != incident_pair_postgamma.shape:
        raise ValueError("unary_postgamma and incident_pair_postgamma shape mismatch")
    if unary_postgamma.shape[-1] != PUBLIC_SLOT_COUNT:
        raise ValueError("P14 accepts exactly 20 public slots and rejects a background slot")
    if unary_postgamma.shape[1] not in (4, 21):
        raise ValueError("P14 targets must be K=4 actions or K=21 reasons")
    if unary_postgamma.device != incident_pair_postgamma.device:
        raise ValueError("contribution tensors must share a device")
    if not torch.is_floating_point(unary_postgamma) or not torch.is_floating_point(incident_pair_postgamma):
        raise TypeError("contribution tensors must be floating point")
    _require_finite("unary_postgamma", unary_postgamma)
    _require_finite("incident_pair_postgamma", incident_pair_postgamma)


def analytical_deletion_deltas(
    *,
    unary_postgamma: Tensor,
    incident_pair_postgamma: Tensor,
) -> dict[str, Tensor | dict[str, Tensor]]:
    """Return exact public-slot deletion deltas without rerunning P9/P10.

    For every target-slot pair this is exactly the unary contribution plus all
    pair contributions incident to that slot.  P10's ``incident_postgamma``
    already counts each unordered pair at both endpoints, so no adjacency or
    background term is introduced here.
    """

    _require_public_contributions(unary_postgamma, incident_pair_postgamma)
    deletion_delta = unary_postgamma.float() + incident_pair_postgamma.float()
    detached = deletion_delta.detach()
    diagnostics = {
        "positive_deletion_mean": detached.clamp_min(0.0).mean(dim=-1),
        "positive_deletion_mass": detached.clamp_min(0.0).sum(dim=-1),
        "negative_deletion_mean": (-detached).clamp_min(0.0).mean(dim=-1),
        "negative_deletion_mass": (-detached).clamp_min(0.0).sum(dim=-1),
        "deletion_rms": detached.square().mean(dim=-1).sqrt(),
    }
    return {
        "deletion_delta": deletion_delta.to(dtype=unary_postgamma.dtype),
        "diagnostics": diagnostics,
    }


def _require_field_and_mask(shared_field: Tensor, region_mask: Tensor, *, validate_values: bool = True) -> None:
    if shared_field.ndim != 4 or shared_field.shape[0] < 1 or shared_field.shape[1] < 1:
        raise ValueError("shared_field must be [B,D,H,W]")
    if region_mask.shape != (shared_field.shape[0], shared_field.shape[2], shared_field.shape[3]):
        raise ValueError("region_mask must be [B,H,W] aligned with shared_field")
    if not torch.is_floating_point(shared_field) or not torch.is_floating_point(region_mask):
        raise TypeError("shared_field and region_mask must be floating point")
    if shared_field.device != region_mask.device:
        raise ValueError("shared_field and region_mask must share a device")
    if validate_values:
        _require_finite("shared_field", shared_field)
        _require_finite("region_mask", region_mask)


def neighborhood_background_mean(
    shared_field: Tensor,
    region_mask: Tensor,
    *,
    validate_values: bool = True,
) -> tuple[Tensor, Tensor]:
    """Compute a detached-selection local background mean in fp32.

    The mean is over the one-patch 8-neighborhood around a hard support.  Empty
    or full supports have no valid background neighborhood and are explicitly
    reported unavailable rather than receiving a zero replacement.
    """

    _require_field_and_mask(shared_field, region_mask, validate_values=validate_values)
    support = (region_mask.detach() > MASK_THRESHOLD)
    dilated = F.max_pool2d(support.float().unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1) > 0.0
    neighborhood = dilated & ~support
    count = neighborhood.float().sum(dim=(-1, -2))
    available = count > 0.0
    weights = neighborhood.unsqueeze(1).to(dtype=torch.float32)
    field_fp32 = shared_field.float()
    # Scale before reduction: summing 1e38-valued patches first overflows
    # fp32 even though their mean is finite.
    normalized_weights = weights / count.clamp_min(1.0).view(-1, 1, 1, 1)
    local_mean = (field_fp32 * normalized_weights).sum(dim=(-1, -2), keepdim=True)
    # A nonzero fallback is never used for unavailable cases, but avoids an
    # accidental zero-deletion interpretation in standalone callers.
    global_weight = 1.0 / float(shared_field.shape[-1] * shared_field.shape[-2])
    fallback = (field_fp32 * global_weight).sum(dim=(-1, -2), keepdim=True)
    local_mean = torch.where(available.view(-1, 1, 1, 1), local_mean, fallback)
    return local_mean.to(dtype=shared_field.dtype), available.detach()


def replace_region_with_neighbor_mean(shared_field: Tensor, region_mask: Tensor, replacement: Tensor) -> Tensor:
    """Return an out-of-place local replacement with no mutation of ``shared_field``."""

    _require_field_and_mask(shared_field, region_mask)
    if replacement.shape != (shared_field.shape[0], shared_field.shape[1], 1, 1):
        raise ValueError("replacement must be [B,D,1,1]")
    if replacement.device != shared_field.device or replacement.dtype != shared_field.dtype:
        raise ValueError("replacement must match shared_field dtype and device")
    _require_finite("replacement", replacement)
    support = (region_mask.detach() > MASK_THRESHOLD).unsqueeze(1)
    return torch.where(support, replacement.expand_as(shared_field), shared_field)


def select_equal_mass_control(
    *,
    slot_masks: Tensor,
    sector_probs: Tensor,
    sample_index: int,
    selected_slot: int | Tensor,
    vertical_tolerance: float = VERTICAL_TOLERANCE,
    mass_tolerance: float = MASS_TOLERANCE,
    max_overlap: float = MAX_OVERLAP,
    validate_values: bool = True,
) -> dict[str, Any]:
    """Vector-rank all 19 public candidates under the four control rules.

    The result intentionally keeps availability and the selected control slot as
    scalar tensors.  ``run_feature_intervention`` combines this availability
    with both neighborhood checks before its one permitted host synchronization.
    """

    if slot_masks.ndim != 4 or slot_masks.shape[1] != PUBLIC_SLOT_COUNT:
        raise ValueError("slot_masks must be [B,20,H,W] with no background slot")
    if sector_probs.shape != (slot_masks.shape[0], PUBLIC_SLOT_COUNT, 3):
        raise ValueError("sector_probs must be [B,20,3]")
    if slot_masks.device != sector_probs.device:
        raise ValueError("slot_masks and sector_probs must share a device")
    if not torch.is_floating_point(slot_masks) or not torch.is_floating_point(sector_probs):
        raise TypeError("slot_masks and sector_probs must be floating point")
    if validate_values:
        _require_finite("slot_masks", slot_masks)
        _require_finite("sector_probs", sector_probs)
    if isinstance(sample_index, bool) or not isinstance(sample_index, numbers.Integral):
        raise ValueError("sample_index must be a public integer")
    sample_index = int(sample_index)
    if not 0 <= sample_index < slot_masks.shape[0]:
        raise ValueError("sample_index must be public in-range")
    if isinstance(selected_slot, Tensor):
        if selected_slot.numel() != 1 or selected_slot.device != slot_masks.device:
            raise ValueError("selected_slot tensor must be scalar and on the slot-mask device")
        selected_slot_tensor = selected_slot.detach().to(dtype=torch.long).reshape(())
    else:
        if isinstance(selected_slot, bool) or not isinstance(selected_slot, numbers.Integral):
            raise ValueError("selected_slot must be a public integer or scalar tensor")
        if not 0 <= selected_slot < PUBLIC_SLOT_COUNT:
            raise ValueError("selected_slot must be public in-range")
        selected_slot_tensor = torch.tensor(int(selected_slot), device=slot_masks.device, dtype=torch.long)
    if not (0.0 < vertical_tolerance <= 1.0 and 0.0 <= mass_tolerance < 1.0 and 0.0 <= max_overlap < 1.0):
        raise ValueError("control-selection tolerances are invalid")

    masks = slot_masks[sample_index].detach().float().clamp(0.0, 1.0)
    sectors = sector_probs[sample_index].detach()
    height = masks.shape[-2]
    candidate_slots = torch.arange(PUBLIC_SLOT_COUNT, device=masks.device)
    y_coordinates = torch.linspace(0.0, 1.0, height, device=masks.device, dtype=masks.dtype).view(1, height, 1)
    masses = masks.sum(dim=(-1, -2))
    mass_epsilon = torch.finfo(masks.dtype).eps
    verticals = (masks * y_coordinates).sum(dim=(-1, -2)) / masses.clamp_min(mass_epsilon)
    selected_index = selected_slot_tensor.reshape(1)
    selected_mask = masks.index_select(0, selected_index).squeeze(0)
    selected_mass = masses.gather(0, selected_index).squeeze(0)
    selected_vertical = verticals.gather(0, selected_index).squeeze(0)
    selected_sector = sectors.index_select(0, selected_index).squeeze(0).argmax()
    intersection = torch.minimum(masks, selected_mask.unsqueeze(0)).sum(dim=(-1, -2))
    union = torch.maximum(masks, selected_mask.unsqueeze(0)).sum(dim=(-1, -2))
    overlaps = torch.where(union > mass_epsilon, intersection / union.clamp_min(mass_epsilon), torch.zeros_like(union))
    vertical_distances = (verticals - selected_vertical).abs()
    mass_ratios = masses / selected_mass.clamp_min(mass_epsilon)
    mass_relative_difference = (masses - selected_mass).abs() / selected_mass.clamp_min(mass_epsilon)
    # This scales with the represented mass, accepting mathematical boundary
    # cases such as 19/20 while still rejecting a true >5% deviation.
    tolerance_slack = (
        32.0
        * torch.finfo(masks.dtype).eps
        * torch.maximum(masses, selected_mass).clamp_min(1.0)
        / selected_mass.clamp_min(1.0)
    )
    sector_matches = sectors.argmax(dim=-1).eq(selected_sector)
    valid = (
        candidate_slots.ne(selected_slot_tensor)
        & (masses > mass_epsilon)
        & sector_matches
        & (vertical_distances <= vertical_tolerance)
        & (mass_relative_difference <= mass_tolerance + tolerance_slack)
        & (overlaps < max_overlap)
    )
    selected_has_support = selected_mass > mass_epsilon
    available = selected_has_support & valid.any()
    metadata: dict[str, Any] = {
        "available": available.detach(),
        "reason": "tensorized_availability",
        "selected_mass": selected_mass.detach(),
        "selected_vertical": selected_vertical.detach(),
        "selected_sector": selected_sector.detach(),
        "vertical_tolerance": float(vertical_tolerance),
        "mass_tolerance": float(mass_tolerance),
        "max_overlap": float(max_overlap),
        "mask_threshold": float(MASK_THRESHOLD),
        "sector_match": torch.zeros((), device=masks.device, dtype=torch.bool),
        "candidate_valid": valid.detach(),
        "candidate_mass_relative_difference": mass_relative_difference.detach(),
    }
    infinity = torch.full_like(masses, float("inf"))
    # Stable sorts compose the required lexicographic order:
    # (mass relative difference, vertical distance, overlap, slot id).
    order = candidate_slots
    for key in (
        torch.where(valid, overlaps, infinity),
        torch.where(valid, vertical_distances, infinity),
        torch.where(valid, mass_relative_difference, infinity),
    ):
        order = order[torch.argsort(key[order], stable=True)]
    chosen = order.select(0, 0)
    chosen_index = chosen.reshape(1)
    metadata.update(
        {
            "control_slot": chosen.detach(),
            "sector_match": sector_matches.gather(0, chosen_index).squeeze(0).detach(),
            "vertical_distance": vertical_distances.gather(0, chosen_index).squeeze(0).detach(),
            "mass_ratio": mass_ratios.gather(0, chosen_index).squeeze(0).detach(),
            "overlap": overlaps.gather(0, chosen_index).squeeze(0).detach(),
        }
    )
    return metadata


def _require_intervention_inputs(
    *,
    shared_field: Tensor,
    slot_masks: Tensor,
    sector_probs: Tensor,
    base_logits: Tensor,
    analytical_deletion: Tensor,
    case_ids: Sequence[str | int],
) -> None:
    if shared_field.ndim != 4 or shared_field.shape[0] < 1:
        raise ValueError("shared_field must be [B,D,H,W]")
    batch, _, height, width = shared_field.shape
    if slot_masks.shape != (batch, PUBLIC_SLOT_COUNT, height, width):
        raise ValueError("slot_masks must be [B,20,H,W] exactly aligned with shared_field")
    if sector_probs.shape != (batch, PUBLIC_SLOT_COUNT, 3):
        raise ValueError("sector_probs must be [B,20,3]")
    if base_logits.ndim != 2 or base_logits.shape[0] != batch or base_logits.shape[1] not in (4, 21):
        raise ValueError("base_logits must be [B,4] or [B,21]")
    if analytical_deletion.shape != (batch, base_logits.shape[1], PUBLIC_SLOT_COUNT):
        raise ValueError("analytical_deletion must be [B,K,20 public slots]")
    for name, value in {
        "shared_field": shared_field,
        "slot_masks": slot_masks,
        "sector_probs": sector_probs,
        "base_logits": base_logits,
        "analytical_deletion": analytical_deletion,
    }.items():
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must be floating point")
        if value.device != shared_field.device:
            raise ValueError(f"{name} must share shared_field's device")


def _require_active_finite_inputs(
    *,
    shared_field: Tensor,
    slot_masks: Tensor,
    sector_probs: Tensor,
    base_logits: Tensor,
    analytical_deletion: Tensor,
) -> None:
    _assert_finite_async("shared_field", shared_field)
    _assert_finite_async("slot_masks", slot_masks)
    _assert_finite_async("sector_probs", sector_probs)
    _assert_finite_async("base_logits", base_logits)
    _assert_finite_async("analytical_deletion", analytical_deletion)


def _unavailable(
    *,
    reason: str,
    optimizer_update: int,
    device: torch.device,
    dtype: torch.dtype,
    margin: float,
    selection: dict[str, Tensor] | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "case_id": None,
        "effects": None,
        "loss": None,
        "loss_terms": None,
        "selection": selection,
        "diagnostics": _diagnostics(
            device=device,
            dtype=dtype,
            optimizer_update=optimizer_update,
            available=False,
            computed=False,
            margin=margin,
        ),
    }


def _callback_logits(callback: Callable[[Tensor], Tensor], field: Tensor, *, name: str, expected_shape: tuple[int, int]) -> Tensor:
    logits = callback(field)
    if not isinstance(logits, Tensor) or logits.shape != expected_shape:
        raise ValueError(f"{name} must return logits shaped {expected_shape}")
    if logits.device != field.device or not torch.is_floating_point(logits):
        raise ValueError(f"{name} must return floating logits on the shared-field device")
    _assert_finite_async(name, logits)
    return logits


def counterfactual_margin_loss(
    *,
    d_selected: Tensor,
    d_control: Tensor,
    d_target: Tensor,
    d_wrong: Tensor,
    margin: float,
) -> dict[str, Tensor]:
    """Exact P14 margin objective with no sign or absolute-value shortcuts."""

    if not isinstance(margin, numbers.Real) or not math.isfinite(float(margin)):
        raise ValueError("margin must be finite")
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    values = (d_selected, d_control, d_target, d_wrong)
    if any(not isinstance(value, Tensor) or value.numel() != 1 for value in values):
        raise ValueError("counterfactual effects must be scalar tensors")
    if len({value.device for value in values}) != 1:
        raise ValueError("counterfactual effects must share a device")
    for name, value in zip(("d_selected", "d_control", "d_target", "d_wrong"), values):
        _assert_finite_async(name, value)
    first = torch.relu(float(margin) + d_control.float() - d_selected.float())
    second = torch.relu(float(margin) + d_wrong.float().abs() - d_target.float().abs())
    return {"loss": first + second, "selected_vs_control": first, "target_vs_wrong": second}


def run_feature_intervention(
    *,
    optimizer_update: int,
    shared_field: Tensor,
    slot_masks: Tensor,
    sector_probs: Tensor,
    base_logits: Tensor,
    analytical_deletion: Tensor,
    public_readout: Callable[[Tensor], Tensor],
    public_contribution: Callable[[Tensor], Tensor],
    case_ids: Sequence[str | int],
    margin: float = 0.10,
    every_optimizer_updates: int = COUNTERFACTUAL_EVERY_UPDATES,
) -> dict[str, Any]:
    """Run one deterministic public-field counterfactual, or report unavailable.

    The only re-executed functions are ``public_readout`` and
    ``public_contribution`` on two out-of-place replacements.  This function
    cannot receive raw inputs, an encoder object, or persistence handles by
    API design.
    """

    optimizer_update = _require_positive_update("optimizer_update", optimizer_update)
    every_optimizer_updates = _require_positive_update("every_optimizer_updates", every_optimizer_updates)
    _require_intervention_inputs(
        shared_field=shared_field,
        slot_masks=slot_masks,
        sector_probs=sector_probs,
        base_logits=base_logits,
        analytical_deletion=analytical_deletion,
        case_ids=case_ids,
    )
    canonical_case_ids = _canonical_case_ids(case_ids, batch=shared_field.shape[0])
    if not isinstance(margin, numbers.Real) or not math.isfinite(float(margin)):
        raise ValueError("margin must be finite")
    # Counterfactual supervision evaluates interventions against a fixed
    # reference.  It may update the replay readouts, never the computation
    # that produced the original reference or its shared field.
    base_reference = base_logits.detach()
    replay_field = shared_field.detach()
    if optimizer_update % every_optimizer_updates != 0:
        return _unavailable(
            reason="optimizer_step_gate",
            optimizer_update=optimizer_update,
            device=replay_field.device,
            dtype=replay_field.dtype,
            margin=float(margin),
        )
    _require_active_finite_inputs(
        shared_field=shared_field,
        slot_masks=slot_masks,
        sector_probs=sector_probs,
        base_logits=base_logits,
        analytical_deletion=analytical_deletion,
    )

    batch, _, height, width = replay_field.shape
    targets = base_reference.shape[1]
    # The schedule chooses one stable row per active update.  All target/slot
    # indices stay as detached device tensors until a caller serializes an
    # artifact, avoiding candidate-level GPU-to-host scalar extraction.
    sample_index = ((optimizer_update // every_optimizer_updates) - 1) % batch
    detached_deletion = analytical_deletion.detach().float()
    selection_scores = torch.where(
        torch.isfinite(detached_deletion),
        detached_deletion,
        torch.full_like(detached_deletion, float("-inf")),
    )
    flat_index = selection_scores[sample_index].reshape(-1).argmax()
    target_index = torch.div(flat_index, PUBLIC_SLOT_COUNT, rounding_mode="floor")
    selected_slot = torch.remainder(flat_index, PUBLIC_SLOT_COUNT)
    base_for_selection = torch.nan_to_num(base_reference[sample_index], nan=float("-inf"))
    target_mask = F.one_hot(target_index, num_classes=targets).to(dtype=torch.bool)
    wrong_target_index = base_for_selection.masked_fill(target_mask, float("-inf")).argmax()
    selection = {
        "sample_index": torch.tensor(sample_index, device=replay_field.device, dtype=torch.long),
        "target_index": target_index.detach(),
        "wrong_target_index": wrong_target_index.detach(),
        "selected_slot": selected_slot.detach(),
    }
    control = select_equal_mass_control(
        slot_masks=slot_masks,
        sector_probs=sector_probs,
        sample_index=sample_index,
        selected_slot=selected_slot,
        validate_values=False,
    )
    control_slot = control["control_slot"]
    if not isinstance(control_slot, Tensor):
        raise RuntimeError("tensorized control selection must return a scalar tensor")
    selection["control_slot"] = control_slot.detach()

    sample_slot_masks = slot_masks[sample_index : sample_index + 1]
    selected_mask = sample_slot_masks.index_select(1, selected_slot.reshape(1)).squeeze(1)
    control_mask = sample_slot_masks.index_select(1, control_slot.reshape(1)).squeeze(1)
    selected_mean, selected_available = neighborhood_background_mean(
        replay_field[sample_index : sample_index + 1], selected_mask, validate_values=False
    )
    control_mean, control_available = neighborhood_background_mean(
        replay_field[sample_index : sample_index + 1], control_mask, validate_values=False
    )

    # This is the sole scheduled host synchronization.  It consolidates public
    # input, control, and neighborhood availability before callback replay.
    input_finite = _all_finite(shared_field, slot_masks, sector_probs, base_logits, analytical_deletion)
    status_code = torch.where(
        ~input_finite,
        torch.ones((), device=replay_field.device, dtype=torch.int64),
        torch.where(
            ~control["available"],
            torch.full((), 2, device=replay_field.device, dtype=torch.int64),
            torch.where(
                ~selected_available.reshape(()),
                torch.full((), 3, device=replay_field.device, dtype=torch.int64),
                torch.where(
                    ~control_available.reshape(()),
                    torch.full((), 4, device=replay_field.device, dtype=torch.int64),
                    torch.zeros((), device=replay_field.device, dtype=torch.int64),
                ),
            ),
        ),
    )
    status = int(status_code.item())
    if status != 0:
        if status == 1:
            raise ValueError("scheduled counterfactual inputs must contain only finite values")
        reason = {
            2: "no_eligible_control",
            3: "selected_slot_has_no_neighborhood",
            4: "control_slot_has_no_neighborhood",
        }[status]
        return _unavailable(
            reason=reason,
            optimizer_update=optimizer_update,
            device=replay_field.device,
            dtype=replay_field.dtype,
            margin=float(margin),
            selection=selection,
        )

    batch_selector = torch.arange(batch, device=replay_field.device).view(batch, 1, 1, 1) == sample_index
    selected_support = (
        slot_masks.index_select(1, selected_slot.reshape(1)).squeeze(1).detach() > MASK_THRESHOLD
    ).unsqueeze(1) & batch_selector
    control_support = (
        slot_masks.index_select(1, control_slot.reshape(1)).squeeze(1).detach() > MASK_THRESHOLD
    ).unsqueeze(1) & batch_selector
    selected_field = torch.where(selected_support, selected_mean.expand(batch, -1, height, width), replay_field)
    control_field = torch.where(control_support, control_mean.expand(batch, -1, height, width), replay_field)
    expected_shape = (batch, targets)
    selected_logits = _callback_logits(public_readout, selected_field, name="public_readout", expected_shape=expected_shape)
    selected_logits = selected_logits + _callback_logits(
        public_contribution, selected_field, name="public_contribution", expected_shape=expected_shape
    )
    control_logits = _callback_logits(public_readout, control_field, name="public_readout", expected_shape=expected_shape)
    control_logits = control_logits + _callback_logits(
        public_contribution, control_field, name="public_contribution", expected_shape=expected_shape
    )

    base_row = base_reference[sample_index : sample_index + 1]
    selected_row = selected_logits[sample_index : sample_index + 1]
    control_row = control_logits[sample_index : sample_index + 1]
    target_column = target_index.reshape(1, 1)
    wrong_column = wrong_target_index.reshape(1, 1)
    d_selected = (base_row.gather(1, target_column) - selected_row.gather(1, target_column)).squeeze()
    d_control = (base_row.gather(1, target_column) - control_row.gather(1, target_column)).squeeze()
    d_wrong = (base_row.gather(1, wrong_column) - selected_row.gather(1, wrong_column)).squeeze()
    loss_terms = counterfactual_margin_loss(
        d_selected=d_selected,
        d_control=d_control,
        d_target=d_selected,
        d_wrong=d_wrong,
        margin=margin,
    )
    effects = {
        "d_selected": d_selected,
        "d_control": d_control,
        "d_target": d_selected,
        "d_wrong": d_wrong,
    }
    diagnostics = _diagnostics(
        device=replay_field.device,
        dtype=replay_field.dtype,
        optimizer_update=optimizer_update,
        available=True,
        computed=True,
        margin=float(margin),
        values={
            "selected_effect": d_selected,
            "control_effect": d_control,
            "target_effect": d_selected,
            "wrong_effect": d_wrong,
            "positive_analytical_effect": detached_deletion[sample_index : sample_index + 1]
            .gather(1, target_index.reshape(1, 1, 1).expand(1, 1, PUBLIC_SLOT_COUNT))
            .squeeze(0)
            .squeeze(0)
            .clamp_min(0.0)
            .mean(),
            "negative_analytical_effect": (-detached_deletion[sample_index : sample_index + 1]
            .gather(1, target_index.reshape(1, 1, 1).expand(1, 1, PUBLIC_SLOT_COUNT))
            .squeeze(0)
            .squeeze(0))
            .clamp_min(0.0)
            .mean(),
        },
    )
    return {
        "available": True,
        "reason": "ok",
        "case_id": canonical_case_ids[sample_index],
        "sample_index": selection["sample_index"],
        "target_index": selection["target_index"],
        "wrong_target_index": selection["wrong_target_index"],
        "selected_slot": selection["selected_slot"],
        "control_slot": selection["control_slot"],
        "selection": selection,
        "control": control,
        "effects": effects,
        "loss": loss_terms["loss"],
        "loss_terms": loss_terms,
        "diagnostics": diagnostics,
    }


__all__ = [
    "COUNTERFACTUAL_EVERY_UPDATES",
    "MASS_TOLERANCE",
    "MASK_THRESHOLD",
    "MAX_OVERLAP",
    "PUBLIC_SLOT_COUNT",
    "VERTICAL_TOLERANCE",
    "analytical_deletion_deltas",
    "counterfactual_margin_loss",
    "neighborhood_background_mean",
    "replace_region_with_neighbor_mean",
    "run_feature_intervention",
    "select_equal_mass_control",
]
