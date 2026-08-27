from __future__ import annotations

import torch


EVENT_NAMES = (
    "approach_pressure",
    "crossing_pressure",
    "corridor_entry",
    "corridor_occupancy",
    "future_collision",
    "slowdown_pressure",
    "flow_incoherence",
    "visibility_support",
    "motion_strength",
    "nearest_distance_risk",
    "pair_convergence",
    "temporal_asymmetry",
)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim) / weight.sum(dim).clamp_min(1.0)


def build_action_interaction_events(
    tracks_xy: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    """Build compact physical traffic events for forward/stop/left/right.

    Shared image motion is removed before measuring object motion. The result is
    deliberately low dimensional so a small calibration cohort cannot identify
    clips from flattened track coordinates instead of learning driving events.
    """
    if tracks_xy.ndim != 4 or tracks_xy.shape[-1] != 2:
        raise ValueError("tracks_xy must be [B,T,K,2]")
    if visibility.shape != tracks_xy.shape[:-1]:
        raise ValueError("visibility must be [B,T,K]")
    if tracks_xy.shape[1] < 3:
        raise ValueError("at least three frames are required")

    visible = visibility.bool()
    pair_visible = visible[:, 1:] & visible[:, :-1]
    raw_step = tracks_xy[:, 1:] - tracks_xy[:, :-1]
    shared = torch.nanmedian(
        raw_step.masked_fill(~pair_visible[..., None], float("nan")), dim=2
    ).values.nan_to_num()
    exclusive_step = (raw_step - shared[:, :, None]) * pair_visible[..., None]

    # Reconstruct coordinates in the terminal camera frame while cancelling
    # shared camera translation. This retains object-relative geometry.
    shared_path = torch.cat(
        (torch.zeros_like(shared[:, :1]), shared.cumsum(1)), dim=1
    )
    stabilized = tracks_xy - shared_path[:, :, None]
    velocity = _masked_mean(exclusive_step, pair_visible, dim=1)
    recent = _masked_mean(exclusive_step[:, -3:], pair_visible[:, -3:], dim=1)
    early = _masked_mean(exclusive_step[:, :3], pair_visible[:, :3], dim=1)
    acceleration = recent - early
    final_xy = stabilized[:, -1]
    support = visible.float().mean(1) * pair_visible.float().mean(1)

    ego = final_xy.new_tensor((0.0, 1.0))
    relative = final_xy - ego
    distance = relative.square().sum(-1).sqrt()
    radial = (velocity * relative).sum(-1) / distance.clamp_min(1e-4)
    approach = (-radial).relu()
    crossing = (
        relative[..., 0] * velocity[..., 1]
        - relative[..., 1] * velocity[..., 0]
    ).abs() / distance.clamp_min(1e-4)
    speed = velocity.square().sum(-1).sqrt()
    accel_along_velocity = -(acceleration * velocity).sum(-1) / speed.clamp_min(1e-4)

    horizons = tracks_xy.new_tensor((2.0, 4.0, 8.0))
    future = final_xy[:, :, None] + velocity[:, :, None] * horizons[None, None, :, None]
    future_distance = (future - ego).square().sum(-1).sqrt()
    future_collision = torch.exp(-3.0 * future_distance.amin(-1))

    pair_relative = final_xy[:, :, None] - final_xy[:, None]
    pair_velocity = velocity[:, :, None] - velocity[:, None]
    pair_distance = pair_relative.square().sum(-1).sqrt()
    pair_closing = (
        -(pair_relative * pair_velocity).sum(-1) / pair_distance.clamp_min(1e-4)
    ).relu()
    diagonal = torch.eye(
        tracks_xy.shape[2], device=tracks_xy.device, dtype=torch.bool
    )[None]
    pair_valid = (
        (support > 0)[:, :, None] & (support > 0)[:, None] & ~diagonal
    )
    pair_convergence = (
        pair_closing / (pair_distance + pair_closing + 1e-4)
    ).masked_fill(~pair_valid, 0.0).amax(-1)

    corridor_x = tracks_xy.new_tensor((0.0, 0.0, -0.45, 0.45))
    rows = []
    for target_x in corridor_x:
        final_lateral = (final_xy[..., 0] - target_x).abs()
        future_lateral = (future[..., 0] - target_x).abs().amin(-1)
        road_weight = torch.sigmoid(4.0 * (final_xy[..., 1] + 0.15))
        corridor_weight = torch.exp(-4.0 * future_lateral) * road_weight * support
        denominator = corridor_weight.sum(-1).clamp_min(1e-6)

        def pool(value: torch.Tensor) -> torch.Tensor:
            return (corridor_weight * value).sum(-1) / denominator

        occupancy = torch.exp(-5.0 * final_lateral)
        entry = (final_lateral - future_lateral).relu().clamp_max(1.0)
        visible_tracks = (support > 0).to(tracks_xy.dtype).sum(-1)
        event = torch.stack(
            (
                pool(approach / (distance + approach + 1e-4)),
                pool(crossing / (distance + crossing + 1e-4)),
                pool(entry),
                pool(occupancy),
                pool(future_collision),
                pool(accel_along_velocity.relu() / (speed + accel_along_velocity.relu() + 1e-4)),
                (exclusive_step.std(2).square().sum(-1).sqrt() * pair_visible.any(2)).mean(1).tanh(),
                (visible_tracks / max(1, tracks_xy.shape[2])).clamp(0.0, 1.0),
                pool(speed.tanh()),
                pool(torch.exp(-2.0 * distance)),
                pool(pair_convergence),
                pool((recent - early).square().sum(-1).sqrt().tanh()),
            ),
            dim=-1,
        ).clamp(0.0, 1.0)
        rows.append(event)
    return torch.stack(rows, dim=1)

