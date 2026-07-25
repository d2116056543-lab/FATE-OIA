"""P6 slot attributes, reliable absence, and conditional grounding losses.

This module intentionally owns only attribute heads and grounding-loss logic.
The P5 ledger remains the owner of road identities and pixel masks; callers
provide its five road masks and twelve entity masks here as image-derived
evidence.  Every weak-label mask is explicit so unavailable BDD100K evidence
cannot silently become a negative target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:  # Keep the P1 type dependency compile-only for lightweight tests.
    from fate_oia.datasets.rael_grounding_targets import EntityGroundingTargets


ENTITY_TYPES = (
    "vehicle",
    "pedestrian",
    "rider",
    "traffic_control",
    "traffic_sign",
    "other",
)
TRAFFIC_STATES = ("red", "green", "yellow_or_other", "unknown")
BOUNDARY_STYLES = ("solid", "dashed_or_other", "unknown")
HORIZONTAL_SECTORS = ("left", "center", "right")
DEPTH_SECTORS = ("near", "middle", "far")
ROAD_MIRROR_PERMUTATION = (2, 1, 0, 4, 3)


@dataclass(frozen=True)
class GroundingLossResult:
    """A real zero-gradient loss with an explicit activation state.

    The zero-valued tensor only preserves the optimizer graph for batches
    without labels.  ``active=False`` is the authoritative signal and must be
    recorded by the trainer rather than reported as a measured zero loss.
    """

    loss: Tensor
    active: bool
    valid_count: int
    components: Mapping[str, Tensor]


def _parameter_dtype(module: nn.Module) -> torch.dtype:
    return next(module.parameters()).dtype


def _as_parameter_dtype(value: Tensor, module: nn.Module) -> Tensor:
    return value.to(dtype=_parameter_dtype(module))


def _require_finite(name: str, value: Tensor) -> None:
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite Tensor")


def _require_probability(name: str, value: Tensor) -> None:
    _require_finite(name, value)
    if bool((value < 0.0).any() or (value > 1.0).any()):
        raise ValueError(f"{name} must lie in [0,1]")


def _zero_result(reference: Tensor, component_names: tuple[str, ...]) -> GroundingLossResult:
    zero = reference.sum() * 0.0
    return GroundingLossResult(
        loss=zero,
        active=False,
        valid_count=0,
        components={name: zero.detach() for name in component_names},
    )


def _valid_count(mask: Tensor) -> int:
    if mask.dtype is not torch.bool:
        raise TypeError("valid masks must be bool")
    return int(mask.sum().item())


def _masked_weighted_mean(values: Tensor, valid: Tensor, reliability: Tensor | None = None) -> Tensor:
    if values.shape != valid.shape:
        raise ValueError("values and valid masks must have identical shapes")
    weights = valid.to(dtype=values.dtype)
    if reliability is not None:
        if reliability.shape != values.shape:
            raise ValueError("reliability must match values")
        _require_probability("reliability", reliability)
        # Reliability is a supervision weight, never a trainable escape path.
        weights = weights * reliability.detach().to(dtype=values.dtype)
    return (values * weights).sum() / valid.to(dtype=values.dtype).sum().clamp_min(1.0)


def _expand_valid(valid: Tensor, reference: Tensor) -> Tensor:
    if valid.dtype is not torch.bool or valid.ndim != 2 or reference.ndim != 4:
        raise ValueError("valid must be bool [B,C] for a [B,C,H,W] tensor")
    if valid.shape != reference.shape[:2]:
        raise ValueError("valid mask and image logits disagree")
    return valid.unsqueeze(-1).unsqueeze(-1).expand_as(reference)


def entity_reliability(observability: Tensor, q_ground: Tensor, q_view: Tensor, q_state: Tensor) -> Tensor:
    """Return rho=o*q_ground*q_view*q_state without presence coupling."""

    if not (observability.shape == q_ground.shape == q_view.shape == q_state.shape):
        raise ValueError("all entity reliability inputs must share [B,J]")
    for name, value in (
        ("observability", observability),
        ("q_ground", q_ground),
        ("q_view", q_view),
        ("q_state", q_state),
    ):
        _require_probability(name, value)
    return observability * q_ground * q_view * q_state


def reliable_absence_evidence(
    entity_presence: Tensor,
    horizontal_sector_probs: Tensor,
    sector_visibility: Tensor,
    q_view: Tensor,
) -> dict[str, Tensor]:
    """Compute exact sector occupancy, clear evidence, and detached rho_clear."""

    if entity_presence.ndim != 2 or horizontal_sector_probs.ndim != 3:
        raise ValueError("entity presence must be [B,J] and sector probabilities [B,J,3]")
    if horizontal_sector_probs.shape[:2] != entity_presence.shape or horizontal_sector_probs.shape[-1] != 3:
        raise ValueError("horizontal sector probabilities must align with entity presence")
    if sector_visibility.shape != (entity_presence.shape[0], 3) or q_view.shape != sector_visibility.shape:
        raise ValueError("visibility and q_view must be [B,3]")
    _require_probability("entity_presence", entity_presence)
    _require_probability("horizontal_sector_probs", horizontal_sector_probs)
    _require_probability("sector_visibility", sector_visibility)
    _require_probability("q_view", q_view)
    if not torch.allclose(
        horizontal_sector_probs.sum(dim=-1),
        torch.ones_like(entity_presence),
        atol=1.0e-5,
    ):
        raise ValueError("horizontal sector probabilities must sum to one")

    terms = (entity_presence.unsqueeze(-1) * horizontal_sector_probs).clamp(0.0, 1.0)
    occupied = 1.0 - torch.prod(1.0 - terms, dim=1)
    clear = sector_visibility * (1.0 - occupied)
    clear_reliability = (sector_visibility * q_view).detach()
    return {
        "left_occupied": occupied[:, 0],
        "center_occupied": occupied[:, 1],
        "right_occupied": occupied[:, 2],
        "left_clear": clear[:, 0],
        "center_clear": clear[:, 1],
        "right_clear": clear[:, 2],
        "occupied": occupied,
        "clear": clear,
        "visibility": sector_visibility,
        "clear_reliability": clear_reliability,
    }


class RAELSlotAttributeHeads(nn.Module):
    """P6 attribute heads over P5-owned entity and road evidence.

    The mask-derived sector logits carry the primary signal.  Token MLPs are
    intentionally restricted to a small additive scale, so a token-only route
    cannot erase left/right or near/far geometry supplied by the ledger.
    """

    parameter_owner = "slot_attribute_heads"
    entity_types = ENTITY_TYPES
    traffic_states = TRAFFIC_STATES
    boundary_styles = BOUNDARY_STYLES
    road_slot_names = (
        "drivable_left",
        "drivable_center",
        "drivable_right",
        "boundary_left",
        "boundary_right",
    )

    def __init__(
        self,
        dim: int = 384,
        *,
        num_entity_slots: int = 12,
        sector_aux_scale: float = 0.05,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_entity_slots <= 0:
            raise ValueError("dim and num_entity_slots must be positive")
        if not 0.0 <= sector_aux_scale <= 0.10:
            raise ValueError("sector_aux_scale must keep MLP assistance weak")
        self.dim = int(dim)
        self.num_entity_slots = int(num_entity_slots)
        self.sector_aux_scale = float(sector_aux_scale)
        self.eps = float(eps)

        self.presence_head = nn.Linear(dim, 1)
        self.observability_head = nn.Linear(dim, 1)
        self.entity_type_head = nn.Linear(dim, len(ENTITY_TYPES))
        self.traffic_state_head = nn.Linear(dim, len(TRAFFIC_STATES))
        self.horizontal_sector_aux = nn.Linear(dim, len(HORIZONTAL_SECTORS))
        self.depth_sector_aux = nn.Linear(dim, len(DEPTH_SECTORS))
        self.drivable_token_head = nn.Linear(dim, 1)
        self.boundary_style_head = nn.Linear(dim, len(BOUNDARY_STYLES))
        self.visibility_head = nn.Linear(dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _sector_geometry(self, masks: Tensor) -> tuple[Tensor, Tensor]:
        if masks.ndim != 4 or masks.shape[1] != self.num_entity_slots:
            raise ValueError("entity masks must be [B,num_entity_slots,H,W]")
        _require_probability("entity_masks", masks)
        _, _, height, width = masks.shape
        x_bins = torch.arange(width, device=masks.device) * 3 // width
        # In driving imagery, rows closest to the camera are at the bottom.
        # Keep the public near/middle/far order while reversing image-row bins.
        y_bins = 2 - (torch.arange(height, device=masks.device) * 3 // height)
        horizontal_mass = torch.stack(
            [masks[..., x_bins == index].sum(dim=(-1, -2)) for index in range(3)], dim=-1
        )
        depth_mass = torch.stack(
            [masks[..., y_bins == index, :].sum(dim=(-1, -2)) for index in range(3)], dim=-1
        )
        horizontal = horizontal_mass / horizontal_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        depth = depth_mass / depth_mass.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        uniform_h = torch.full_like(horizontal, 1.0 / 3.0)
        uniform_d = torch.full_like(depth, 1.0 / 3.0)
        horizontal = torch.where(horizontal_mass.sum(dim=-1, keepdim=True) > self.eps, horizontal, uniform_h)
        depth = torch.where(depth_mass.sum(dim=-1, keepdim=True) > self.eps, depth, uniform_d)
        return horizontal, depth

    def forward(
        self,
        entity_tokens: Tensor,
        entity_masks: Tensor,
        road_tokens: Tensor,
        road_masks: Tensor,
        *,
        q_ground: Tensor | None = None,
        q_view: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if entity_tokens.ndim != 3 or entity_tokens.shape[1:] != (self.num_entity_slots, self.dim):
            raise ValueError("entity tokens must be [B,num_entity_slots,D]")
        if road_tokens.ndim != 3 or road_tokens.shape[1:] != (5, self.dim):
            raise ValueError("road tokens must be [B,5,D]")
        if road_masks.ndim != 4 or road_masks.shape[:2] != (entity_tokens.shape[0], 5):
            raise ValueError("road masks must be [B,5,H,W]")
        if entity_masks.shape[0] != entity_tokens.shape[0] or entity_masks.shape[-2:] != road_masks.shape[-2:]:
            raise ValueError("entity/road masks must share batch and geometry")
        _require_finite("entity_tokens", entity_tokens)
        _require_finite("road_tokens", road_tokens)
        _require_probability("road_masks", road_masks)

        entity_tokens = _as_parameter_dtype(entity_tokens, self)
        road_tokens = _as_parameter_dtype(road_tokens, self)
        entity_masks = entity_masks.to(dtype=entity_tokens.dtype)
        road_masks = road_masks.to(dtype=entity_tokens.dtype)
        batch = entity_tokens.shape[0]
        horizontal_geometry, depth_geometry = self._sector_geometry(entity_masks)
        horizontal_logits = torch.log(horizontal_geometry.clamp_min(self.eps)) + self.sector_aux_scale * self.horizontal_sector_aux(entity_tokens)
        depth_logits = torch.log(depth_geometry.clamp_min(self.eps)) + self.sector_aux_scale * self.depth_sector_aux(entity_tokens)
        horizontal_probs = torch.softmax(horizontal_logits, dim=-1)
        depth_probs = torch.softmax(depth_logits, dim=-1)

        entity_type_logits = self.entity_type_head(entity_tokens)
        entity_type_probs = torch.softmax(entity_type_logits, dim=-1)
        traffic_state_logits = self.traffic_state_head(entity_tokens)
        traffic_state_probs = torch.softmax(traffic_state_logits, dim=-1)
        presence_logits = self.presence_head(entity_tokens).squeeze(-1)
        observability_logits = self.observability_head(entity_tokens).squeeze(-1)
        presence = torch.sigmoid(presence_logits)
        observability = torch.sigmoid(observability_logits)

        # Road masks are P5-owned evidence.  This head only maps the fixed
        # identity token into a bounded pixel-wise confidence adjustment.
        mask_logit = torch.logit(road_masks[:, :3].clamp(self.eps, 1.0 - self.eps))
        drivable_logits = mask_logit + self.drivable_token_head(road_tokens[:, :3]).view(batch, 3, 1, 1)
        boundary_style_logits = self.boundary_style_head(road_tokens[:, 3:5])
        boundary_style_probs = torch.softmax(boundary_style_logits, dim=-1)
        sector_visibility = torch.sigmoid(self.visibility_head(road_tokens[:, :3]).squeeze(-1))

        if q_ground is None:
            q_ground = torch.ones_like(presence)
        if q_view is None:
            q_view = torch.ones_like(presence)
        _require_probability("q_ground", q_ground)
        _require_probability("q_view", q_view)
        q_ground = q_ground.to(dtype=presence.dtype, device=presence.device)
        q_view = q_view.to(dtype=presence.dtype, device=presence.device)
        type_confidence = entity_type_probs.max(dim=-1).values
        state_confidence = traffic_state_probs.max(dim=-1).values
        traffic_probability = entity_type_probs[..., ENTITY_TYPES.index("traffic_control")]
        q_state = type_confidence * ((1.0 - traffic_probability) + traffic_probability * state_confidence)
        reliability = entity_reliability(observability, q_ground, q_view, q_state)

        return {
            "presence_logits": presence_logits,
            "presence": presence,
            "observability_logits": observability_logits,
            "observability": observability,
            "entity_type_logits": entity_type_logits,
            "entity_type_probs": entity_type_probs,
            "traffic_state_logits": traffic_state_logits,
            "traffic_state_probs": traffic_state_probs,
            "horizontal_sector_probs": horizontal_probs,
            "depth_sector_probs": depth_probs,
            "sector_joint_probs": horizontal_probs.unsqueeze(-1) * depth_probs.unsqueeze(-2),
            "drivable_logits": drivable_logits,
            "boundary_style_logits": boundary_style_logits,
            "boundary_style_probs": boundary_style_probs,
            "sector_visibility": sector_visibility,
            "q_ground": q_ground,
            "q_view": q_view,
            "q_state": q_state,
            "entity_reliability": reliability,
        }


def reliability_weighted_bce(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    reliability: Tensor,
) -> GroundingLossResult:
    """BCE with detached reliability and explicit unknown-label masking."""

    if logits.shape != targets.shape or logits.shape != valid_mask.shape or logits.shape != reliability.shape:
        raise ValueError("logits, targets, valid_mask, and reliability must share shape")
    _require_probability("targets", targets)
    _require_probability("reliability", reliability)
    count = _valid_count(valid_mask)
    if count == 0:
        return _zero_result(logits, ("bce",))
    bce = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
    loss = _masked_weighted_mean(bce, valid_mask, reliability)
    return GroundingLossResult(loss=loss, active=True, valid_count=count, components={"bce": loss.detach()})


def _masked_cross_entropy(logits: Tensor, targets: Tensor, valid_mask: Tensor, reliability: Tensor) -> GroundingLossResult:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or valid_mask.shape != targets.shape:
        raise ValueError("classification shapes must be logits [B,J,C], targets/mask [B,J]")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("classification targets must be integer indices")
    if reliability.shape != targets.shape:
        raise ValueError("classification reliability must be [B,J]")
    known = valid_mask & (targets >= 0)
    count = _valid_count(known)
    if count == 0:
        return _zero_result(logits, ("cross_entropy",))
    if bool((targets[known] >= logits.shape[-1]).any()):
        raise ValueError("classification target index out of range")
    # Invalid/unknown rows are masked below, but cross_entropy still receives a
    # dense target tensor.  Clamp both ends so an arbitrary sentinel cannot
    # crash the batch before its mask excludes it.
    safe_targets = targets.clamp(min=0, max=logits.shape[-1] - 1)
    ce = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none")
    loss = _masked_weighted_mean(ce, known, reliability)
    return GroundingLossResult(loss=loss, active=True, valid_count=count, components={"cross_entropy": loss.detach()})


def entity_attribute_grounding_loss(outputs: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> dict[str, GroundingLossResult]:
    """Apply entity attribute losses with traffic-state-only conditioning."""

    required_outputs = ("presence_logits", "entity_type_logits", "traffic_state_logits", "entity_reliability")
    required_targets = ("presence", "presence_valid", "type", "type_valid", "traffic_state", "traffic_state_valid")
    missing = [name for name in (*required_outputs, *required_targets) if name not in (outputs if name in required_outputs else targets)]
    if missing:
        raise KeyError(f"missing entity grounding fields: {missing}")
    reliability = outputs["entity_reliability"]
    presence = reliability_weighted_bce(
        outputs["presence_logits"], targets["presence"], targets["presence_valid"], reliability
    )
    entity_type = _masked_cross_entropy(
        outputs["entity_type_logits"], targets["type"], targets["type_valid"], reliability
    )
    traffic_control = ENTITY_TYPES.index("traffic_control")
    traffic_valid = (
        targets["traffic_state_valid"]
        & targets["type_valid"]
        & targets["type"].eq(traffic_control)
        & targets["traffic_state"].ge(0)
    )
    traffic_state = _masked_cross_entropy(
        outputs["traffic_state_logits"], targets["traffic_state"], traffic_valid, reliability
    )
    return {"presence": presence, "entity_type": entity_type, "traffic_state": traffic_state}


def _coerce_reliability_map(reliability: Tensor | None, reference: Tensor) -> Tensor | None:
    if reliability is None:
        return None
    if reliability.shape != reference.shape[:2]:
        raise ValueError("road reliability must be [B,C]")
    _require_probability("road reliability", reliability)
    return reliability.unsqueeze(-1).unsqueeze(-1).expand_as(reference)


def drivable_bce_dice_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    reliability: Tensor | None = None,
    *,
    eps: float = 1.0e-6,
) -> GroundingLossResult:
    """Conditional BCE+Dice for the three fixed drivable road identities."""

    if logits.ndim != 4 or logits.shape[1] != 3 or targets.shape != logits.shape:
        raise ValueError("drivable logits/targets must be [B,3,H,W]")
    _require_probability("drivable targets", targets)
    valid_pixels = _expand_valid(valid_mask, logits)
    count = _valid_count(valid_mask)
    if count == 0:
        return _zero_result(logits, ("bce", "dice"))
    road_weight = _coerce_reliability_map(reliability, logits)
    bce_values = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
    bce = _masked_weighted_mean(bce_values, valid_pixels, road_weight)
    probs = torch.sigmoid(logits)
    valid_float = valid_mask.to(dtype=logits.dtype)
    if reliability is not None:
        valid_float = valid_float * reliability.detach().to(dtype=logits.dtype)
    intersection = (probs * targets.to(dtype=logits.dtype)).sum(dim=(-1, -2))
    denominator = probs.sum(dim=(-1, -2)) + targets.to(dtype=logits.dtype).sum(dim=(-1, -2))
    dice_each = 1.0 - (2.0 * intersection + eps) / (denominator + eps)
    dice = (dice_each * valid_float).sum() / valid_mask.to(dtype=logits.dtype).sum().clamp_min(1.0)
    loss = 0.5 * (bce + dice)
    return GroundingLossResult(
        loss=loss,
        active=True,
        valid_count=count,
        components={"bce": bce.detach(), "dice": dice.detach()},
    )


def _dilate(binary: Tensor, kernel_size: int = 3) -> Tensor:
    return F.max_pool2d(binary, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


def soft_erode(image: Tensor) -> Tensor:
    """Differentiable morphological erosion used by soft clDice."""

    if image.ndim != 4:
        raise ValueError("soft erosion requires [B,C,H,W]")
    _require_probability("soft erosion input", image)
    vertical = -F.max_pool2d(-image, kernel_size=(3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-image, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_dilate(image: Tensor) -> Tensor:
    """Differentiable morphological dilation used by soft opening."""

    if image.ndim != 4:
        raise ValueError("soft dilation requires [B,C,H,W]")
    _require_probability("soft dilation input", image)
    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_open(image: Tensor) -> Tensor:
    """Soft opening: erosion followed by dilation without hard thresholding."""

    return soft_dilate(soft_erode(image))


def soft_skeletonize(image: Tensor, *, iterations: int = 16) -> Tensor:
    """Iterative soft skeletonization from the clDice formulation.

    The operator uses no detach or threshold on the prediction path.  It is
    therefore able to propagate topology gradients to thin, broken, or shifted
    boundary responses instead of merely measuring overlap after the fact.
    """

    if iterations <= 0:
        raise ValueError("soft skeleton iterations must be positive")
    _require_probability("soft skeleton input", image)
    working = image
    skeleton = F.relu(working - soft_open(working))
    for _ in range(iterations):
        working = soft_erode(working)
        delta = F.relu(working - soft_open(working))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp(0.0, 1.0)


def _distance_grid(height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    y, x = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((y, x), dim=-1).reshape(-1, 2)


def _distance_resolution(probs: Tensor, targets: Tensor, max_side: int) -> tuple[Tensor, Tensor]:
    """Use a bounded, aspect-preserving grid for real two-way distance maps."""

    height, width = probs.shape[-2:]
    if max(height, width) <= max_side:
        return probs, targets.detach()
    scale = float(max_side) / float(max(height, width))
    resized = (max(1, round(height * scale)), max(1, round(width * scale)))
    return (
        F.interpolate(probs, size=resized, mode="bilinear", align_corners=False),
        F.interpolate(targets.detach(), size=resized, mode="nearest"),
    )


def _soft_distance_to_support(
    support: Tensor,
    *,
    temperature: float,
    chunk_size: int,
    eps: float,
) -> Tensor:
    """Differentiable soft distance transform to a soft support image.

    Distances are evaluated in chunks to avoid a persistent `[B,C,N,N]`
    allocation.  The support path stays differentiable, while callers pass
    detached supervision masks for target geometry.
    """

    if support.ndim != 4:
        raise ValueError("distance support must be [B,C,H,W]")
    if temperature <= 0.0 or chunk_size <= 0:
        raise ValueError("distance temperature and chunk size must be positive")
    _require_probability("distance support", support)
    batch, channels, height, width = support.shape
    grid = _distance_grid(height, width, device=support.device, dtype=support.dtype)
    log_support = torch.log(support.reshape(batch, channels, -1).clamp_min(eps))
    chunks: list[Tensor] = []
    for start in range(0, grid.shape[0], chunk_size):
        query = grid[start : start + chunk_size]
        pairwise = torch.cdist(query.unsqueeze(0), grid.unsqueeze(0)).squeeze(0)
        distances = pairwise.view(1, 1, query.shape[0], -1)
        scores = log_support.unsqueeze(-2) - distances / temperature
        # A support-weighted soft argmin is a proper non-negative distance at
        # every resolution.  Unlike raw log-sum-exp, it cannot collapse a
        # dense uncertain map to a negative value that is then clamped to zero.
        weights = torch.softmax(scores, dim=-1)
        chunks.append((weights * distances).sum(dim=-1))
    return torch.cat(chunks, dim=-1).reshape(batch, channels, height, width)


def _reduce_channel_values(values: Tensor, valid_mask: Tensor, reliability: Tensor | None) -> Tensor:
    if values.shape != valid_mask.shape:
        raise ValueError("channel values and valid mask must have identical [B,C] shape")
    weights = valid_mask.to(dtype=values.dtype)
    if reliability is not None:
        if reliability.shape != values.shape:
            raise ValueError("distance reliability must be [B,C]")
        _require_probability("distance reliability", reliability)
        weights = weights * reliability.detach().to(dtype=values.dtype)
    return (values * weights).sum() / valid_mask.to(dtype=values.dtype).sum().clamp_min(1.0)


def symmetric_distance_transform_loss(
    probs: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    reliability: Tensor | None = None,
    *,
    temperature: float = 0.05,
    max_side: int = 40,
    chunk_size: int = 256,
    eps: float = 1.0e-6,
) -> dict[str, Tensor]:
    """True target-to-prediction plus prediction-to-target soft DT loss.

    `target_to_pred` exposes missing target components even when all predicted
    mass sits on another correct component. `pred_to_target` penalizes extra
    predicted mass away from the labelled boundary.  Both terms are
    differentiable with respect to `probs`; target geometry is detached.
    """

    if probs.ndim != 4 or probs.shape[1] != 2 or targets.shape != probs.shape:
        raise ValueError("distance probabilities/targets must be [B,2,H,W]")
    _require_probability("distance probabilities", probs)
    _require_probability("distance targets", targets)
    count = _valid_count(valid_mask)
    if count == 0:
        zero = probs.sum() * 0.0
        return {"loss": zero, "target_to_pred": zero.detach(), "pred_to_target": zero.detach()}
    if max_side <= 0:
        raise ValueError("max_side must be positive")
    local_probs, local_targets = _distance_resolution(probs, targets, max_side=max_side)
    target_mass = local_targets.sum(dim=(-1, -2))
    has_target = target_mass > eps
    distance_to_prediction = _soft_distance_to_support(
        local_probs, temperature=temperature, chunk_size=chunk_size, eps=eps
    )
    distance_to_target = _soft_distance_to_support(
        local_targets.detach(), temperature=temperature, chunk_size=chunk_size, eps=eps
    )
    target_to_pred_each = (
        (local_targets * distance_to_prediction).sum(dim=(-1, -2)) / target_mass.clamp_min(eps)
    )
    pred_mass = local_probs.sum(dim=(-1, -2))
    pred_to_target_each = (
        (local_probs * distance_to_target).sum(dim=(-1, -2)) / pred_mass.clamp_min(eps)
    )
    # Empty labelled boundary maps are handled by conditional BCE rather than
    # fabricating a geometric point at the image origin.
    target_to_pred_each = torch.where(has_target, target_to_pred_each, torch.zeros_like(target_to_pred_each))
    pred_to_target_each = torch.where(has_target, pred_to_target_each, torch.zeros_like(pred_to_target_each))
    target_to_pred = _reduce_channel_values(target_to_pred_each, valid_mask, reliability)
    pred_to_target = _reduce_channel_values(pred_to_target_each, valid_mask, reliability)
    return {
        "loss": 0.5 * (target_to_pred + pred_to_target),
        "target_to_pred": target_to_pred,
        "pred_to_target": pred_to_target,
    }


def boundary_dilated_bce_cldice_symmetric_distance_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    reliability: Tensor | None = None,
    *,
    eps: float = 1.0e-6,
) -> GroundingLossResult:
    """Conditional dilated BCE + clDice + symmetric distance-transform loss."""

    if logits.ndim != 4 or logits.shape[1] != 2 or targets.shape != logits.shape:
        raise ValueError("boundary logits/targets must be [B,2,H,W]")
    _require_probability("boundary targets", targets)
    valid_pixels = _expand_valid(valid_mask, logits)
    count = _valid_count(valid_mask)
    if count == 0:
        return _zero_result(logits, ("dilated_bce", "cldice", "symmetric_distance"))
    target = targets.to(dtype=logits.dtype)
    dilated_target = _dilate(target)
    road_weight = _coerce_reliability_map(reliability, logits)
    bce_values = F.binary_cross_entropy_with_logits(logits, dilated_target, reduction="none")
    bce = _masked_weighted_mean(bce_values, valid_pixels, road_weight)
    probs = torch.sigmoid(logits)
    prediction_skeleton = soft_skeletonize(probs)
    target_skeleton = soft_skeletonize(target.detach())
    tprec = (prediction_skeleton * target).sum(dim=(-1, -2)) / prediction_skeleton.sum(dim=(-1, -2)).clamp_min(eps)
    tsens = (target_skeleton * probs).sum(dim=(-1, -2)) / target_skeleton.sum(dim=(-1, -2)).clamp_min(eps)
    cldice_each = 1.0 - (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    distance_terms = symmetric_distance_transform_loss(
        probs,
        target,
        valid_mask,
        reliability,
        eps=eps,
    )
    weights = valid_mask.to(dtype=logits.dtype)
    if reliability is not None:
        if reliability.shape != valid_mask.shape:
            raise ValueError("boundary reliability must be [B,2]")
        _require_probability("boundary reliability", reliability)
        weights = weights * reliability.detach().to(dtype=logits.dtype)
    denominator = valid_mask.to(dtype=logits.dtype).sum().clamp_min(1.0)
    cldice = (cldice_each * weights).sum() / denominator
    symmetric = distance_terms["loss"]
    loss = (bce + cldice + symmetric) / 3.0
    return GroundingLossResult(
        loss=loss,
        active=True,
        valid_count=count,
        components={
            "dilated_bce": bce.detach(),
            "topology_precision": _reduce_channel_values(tprec, valid_mask, reliability).detach(),
            "topology_sensitivity": _reduce_channel_values(tsens, valid_mask, reliability).detach(),
            "cldice": cldice.detach(),
            "symmetric_distance": symmetric.detach(),
            "target_to_pred": distance_terms["target_to_pred"].detach(),
            "pred_to_target": distance_terms["pred_to_target"].detach(),
        },
    )


def mirror_sector_and_road_ids(
    horizontal_sector_probs: Tensor,
    road_values: Tensor,
    *,
    depth_sector_probs: Tensor | None = None,
    sector_joint_probs: Tensor | None = None,
) -> dict[str, Tensor]:
    """Mirror all left/right sectors and five fixed road identities.

    Near/middle/far is a depth axis, not a horizontal identity, and remains
    unchanged.  A joint [horizontal,depth] distribution is permuted only over
    its horizontal dimension.
    """

    if horizontal_sector_probs.ndim != 3 or horizontal_sector_probs.shape[-1] != 3:
        raise ValueError("horizontal sectors must be [B,J,3]")
    if road_values.ndim < 3 or road_values.shape[1] != 5:
        raise ValueError("road values must have five fixed identities")
    mirrored_road_values = road_values
    # Spatial road maps live in image coordinates, so a horizontal mirror must
    # flip pixels before swapping the fixed left/right road identities.  Token
    # readouts [B,5,D] have no spatial axis and only require the permutation.
    if road_values.ndim >= 4:
        mirrored_road_values = mirrored_road_values.flip(dims=(-1,))
    result = {
        "horizontal_sector_probs": horizontal_sector_probs.flip(dims=(-1,)),
        "road_values": mirrored_road_values.index_select(
            1, torch.tensor(ROAD_MIRROR_PERMUTATION, device=road_values.device)
        ),
    }
    if depth_sector_probs is not None:
        if depth_sector_probs.shape != horizontal_sector_probs.shape:
            raise ValueError("depth sectors must match horizontal [B,J,3]")
        result["depth_sector_probs"] = depth_sector_probs
    if sector_joint_probs is not None:
        if sector_joint_probs.shape != horizontal_sector_probs.shape + (3,):
            raise ValueError("sector joint probabilities must be [B,J,3,3]")
        result["sector_joint_probs"] = sector_joint_probs.flip(dims=(-2,))
    return result


def entity_attribute_targets_from_p1(
    entity_targets: "EntityGroundingTargets",
    detections: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    device: torch.device | str,
) -> dict[str, Tensor]:
    """Adapt P1 Hungarian targets into conditional P6 attribute tensors.

    This consumes only BDD100K train-time observations.  Missing type/state
    fields remain unknown (``valid=False``), never fabricated negatives.
    """

    objectness = entity_targets.objectness
    slots = len(objectness)
    presence = torch.zeros(1, slots, device=device)
    presence_valid = torch.zeros(1, slots, device=device, dtype=torch.bool)
    type_target = torch.full((1, slots), -1, device=device, dtype=torch.long)
    type_valid = torch.zeros_like(presence_valid)
    state_target = torch.full_like(type_target, -1)
    state_valid = torch.zeros_like(presence_valid)
    q_ground = torch.zeros_like(presence)
    assignments = {item.slot_index: item for item in entity_targets.assignments}
    category_map = {
        "car": "vehicle", "bus": "vehicle", "truck": "vehicle", "vehicle": "vehicle",
        "pedestrian": "pedestrian", "person": "pedestrian", "rider": "rider", "cyclist": "rider",
        "traffic_light": "traffic_control", "traffic_control": "traffic_control",
        "traffic_sign": "traffic_sign", "sign": "traffic_sign",
    }
    for index, target in enumerate(objectness):
        presence[0, index] = float(target.target)
        presence_valid[0, index] = bool(target.reliable)
        if target.reliable:
            q_ground[0, index] = 1.0
        assignment = assignments.get(index)
        if assignment is None or assignment.detection_index >= len(detections):
            continue
        detection = detections[assignment.detection_index]
        raw_category = str(detection.get("category") or detection.get("type") or "other").lower().replace(" ", "_")
        category = category_map.get(raw_category, "other")
        type_target[0, index] = ENTITY_TYPES.index(category)
        type_valid[0, index] = True
        q_ground[0, index] = 1.0 / (1.0 + max(float(assignment.cost), 0.0))
    for traffic in entity_targets.traffic_state_targets:
        if not traffic.valid or traffic.matched_slot_index is None or traffic.state is None:
            continue
        state_target[0, traffic.matched_slot_index] = TRAFFIC_STATES.index(traffic.state)
        state_valid[0, traffic.matched_slot_index] = True
    return {
        "presence": presence,
        "presence_valid": presence_valid,
        "type": type_target,
        "type_valid": type_valid,
        "traffic_state": state_target,
        "traffic_state_valid": state_valid,
        "q_ground": q_ground,
    }


def _normalise_boundary_style(value: Any) -> int | None:
    """Map explicit lane style metadata to the fixed three-class schema."""

    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not text or text in {"unknown", "none", "null", "unlabeled", "unlabelled", "na", "n/a"}:
        return None
    if "solid" in text and "dash" not in text:
        return BOUNDARY_STYLES.index("solid")
    # The plan deliberately merges all explicit non-solid marked styles into
    # the dashed-or-other class.  Absence of an attribute remains unknown.
    return BOUNDARY_STYLES.index("dashed_or_other")


def _normalise_lane_side(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("side", "sector", "road_identity", "identity"):
            if key in value:
                return _normalise_lane_side(value[key])
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        # P1's fixed road identities use left/center/right ordering.
        return {0: "left", 1: None, 2: "right"}.get(value)
    text = str(value).strip().lower() if value is not None else ""
    if text in {"left", "right"}:
        return text
    # Center/unknown lanes do not supervise either fixed boundary style head.
    if text in {"center", "unknown", "none", "unassigned", ""}:
        return None
    return None


def _p1_lane_assignment(lane_index: int, assignments: Any) -> tuple[bool, str | None]:
    """Return whether P1 assigned this lane and its fixed-third side, if any."""

    assigned: Any = None
    found = False
    if isinstance(assignments, Mapping):
        if lane_index in assignments:
            assigned = assignments[lane_index]
            found = True
        elif str(lane_index) in assignments:
            assigned = assignments[str(lane_index)]
            found = True
    elif assignments is not None:
        for item in assignments:
            if isinstance(item, Mapping) and int(item.get("lane_index", item.get("index", -1))) == lane_index:
                assigned = item.get("side", item.get("sector"))
                found = True
                break
            if isinstance(item, tuple) and len(item) >= 2 and int(item[0]) == lane_index:
                assigned = item[1]
                found = True
                break
    return found, _normalise_lane_side(assigned)


def _lane_side(lane: Mapping[str, Any], lane_index: int, assignments: Any, image_width: float) -> str | None:
    assigned, p1_side = _p1_lane_assignment(lane_index, assignments)
    # An explicit P1 assignment is authoritative.  In particular, center and
    # unknown assignments must not be overwritten by lane metadata or geometry.
    if assigned:
        return p1_side
    for candidate in (
        lane.get("side"),
        lane.get("sector"),
        (lane.get("attributes") or {}).get("side") if isinstance(lane.get("attributes"), Mapping) else None,
    ):
        side = _normalise_lane_side(candidate)
        if side is not None:
            return side
        if candidate is not None:
            return None
    points = lane.get("points", lane.get("poly2d"))
    coordinates: list[tuple[float, float]] = []
    if isinstance(points, (tuple, list)):
        for point in points:
            if isinstance(point, Mapping) and "x" in point and "y" in point:
                coordinates.append((float(point["x"]), float(point["y"])))
            elif isinstance(point, (tuple, list)) and len(point) >= 2:
                coordinates.append((float(point[0]), float(point[1])))
    if not coordinates:
        return None
    mean_x = sum(point[0] for point in coordinates) / len(coordinates)
    # P1 road identity is left/center/right thirds, not a binary midline.
    if mean_x < image_width / 3.0:
        return "left"
    if mean_x >= image_width * (2.0 / 3.0):
        return "right"
    return None


def build_boundary_style_targets(
    lanes: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    assignments: Any = None,
    *,
    device: torch.device | str,
    image_width: float = 640.0,
) -> dict[str, Tensor | int | bool]:
    """Build conditional left/right boundary style supervision from lanes.

    A missing/unknown style is never changed into a negative class.  Conflicting
    explicit styles on the same fixed boundary also become unknown because P1
    does not provide a reliable instance identity for choosing one arbitrarily.
    """

    if image_width <= 0.0:
        raise ValueError("image_width must be positive")
    candidates: dict[str, list[int]] = {"left": [], "right": []}
    for index, lane in enumerate(lanes):
        if not isinstance(lane, Mapping):
            raise TypeError("lanes must contain mappings")
        side = _lane_side(lane, index, assignments, image_width)
        if side is None:
            continue
        attributes = lane.get("attributes")
        attributes = attributes if isinstance(attributes, Mapping) else {}
        raw_style = next(
            (
                attributes[key]
                for key in ("lineStyle", "line_style", "boundaryStyle", "boundary_style", "style", "type")
                if key in attributes
            ),
            next(
                (lane[key] for key in ("lineStyle", "line_style", "boundaryStyle", "boundary_style", "style", "type") if key in lane),
                None,
            ),
        )
        style = _normalise_boundary_style(raw_style)
        if style is not None:
            candidates[side].append(style)
    targets = torch.full((1, 2), -1, dtype=torch.long, device=device)
    valid = torch.zeros((1, 2), dtype=torch.bool, device=device)
    for side, output_index in (("left", 0), ("right", 1)):
        unique = set(candidates[side])
        if len(unique) == 1:
            targets[0, output_index] = unique.pop()
            valid[0, output_index] = True
    count = _valid_count(valid)
    return {"targets": targets, "valid_mask": valid, "valid_count": count, "active": count > 0}


def boundary_style_cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor,
    reliability: Tensor | None = None,
) -> GroundingLossResult:
    """Conditional CE for the two fixed road-boundary style heads."""

    if logits.ndim != 3 or logits.shape[1:] != (2, len(BOUNDARY_STYLES)):
        raise ValueError("boundary style logits must be [B,2,3]")
    if targets.shape != logits.shape[:2] or valid_mask.shape != targets.shape:
        raise ValueError("boundary style targets/mask must be [B,2]")
    if reliability is None:
        reliability = torch.ones_like(targets, dtype=logits.dtype)
    return _masked_cross_entropy(logits, targets, valid_mask, reliability)


def road_grounding_loss_bundle(
    *,
    drivable_logits: Tensor,
    drivable_targets: Tensor,
    drivable_valid_mask: Tensor,
    boundary_logits: Tensor,
    boundary_targets: Tensor,
    boundary_valid_mask: Tensor,
    boundary_style_logits: Tensor,
    boundary_style_targets: Tensor,
    boundary_style_valid_mask: Tensor,
    drivable_reliability: Tensor | None = None,
    boundary_reliability: Tensor | None = None,
) -> dict[str, GroundingLossResult]:
    """Return all P6 road losses, including conditional boundary-style CE."""

    return {
        "drivable": drivable_bce_dice_loss(
            drivable_logits, drivable_targets, drivable_valid_mask, drivable_reliability
        ),
        "boundary": boundary_dilated_bce_cldice_symmetric_distance_loss(
            boundary_logits, boundary_targets, boundary_valid_mask, boundary_reliability
        ),
        "boundary_style": boundary_style_cross_entropy_loss(
            boundary_style_logits,
            boundary_style_targets,
            boundary_style_valid_mask,
            boundary_reliability,
        ),
    }


def _owned_named_parameters(module: nn.Module, *, owner: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        named = tuple(module.named_parameters(remove_duplicate=False))
    except TypeError:  # Compatibility with older PyTorch releases.
        named = tuple(module.named_parameters())
    names = tuple(name for name, _ in named)
    parameter_ids = tuple(id(parameter) for _, parameter in named)
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError(f"{owner} contains duplicate parameter ownership")
    return names, parameter_ids


def p6_parameter_ownership(
    heads: RAELSlotAttributeHeads,
    ledger: nn.Module | None = None,
) -> dict[str, object]:
    """Audit P6 heads and the optional real P5 ledger without fake ownership.

    Older callers that only own P6 heads remain supported, but their report is
    explicitly unverifiable for cross-owner uniqueness until the live ledger is
    supplied by the integration path.
    """

    if not isinstance(heads, RAELSlotAttributeHeads):
        raise TypeError("P6 ownership requires RAELSlotAttributeHeads")
    head_names, head_ids = _owned_named_parameters(heads, owner="P6 attribute heads")
    report: dict[str, object] = {
        "slot_attribute_heads": {
            "owner": "slot_attribute_heads",
            "named_parameters": head_names,
            "parameter_ids": head_ids,
        },
        "slot_ledger_core": {
            "owner": "slot_ledger_core",
            "named_parameters": (),
            "parameter_ids": (),
            "readouts": ("entity_masks", "road_masks", "road_slot_ids"),
        },
        "verified": False,
    }
    if ledger is None:
        return report
    if not isinstance(ledger, nn.Module):
        raise TypeError("P6 ownership ledger must be an nn.Module")
    ledger_names, ledger_ids = _owned_named_parameters(ledger, owner="P5 slot ledger")
    if set(head_ids) & set(ledger_ids):
        raise RuntimeError("P6 attribute heads and P5 ledger share parameters")
    report["slot_ledger_core"] = {
        "owner": "slot_ledger_core",
        "named_parameters": ledger_names,
        "parameter_ids": ledger_ids,
        "readouts": ("entity_masks", "road_masks", "road_slot_ids"),
    }
    report["verified"] = True
    return report
