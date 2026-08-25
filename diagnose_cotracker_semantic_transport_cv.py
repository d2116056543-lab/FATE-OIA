from __future__ import annotations

import json
from pathlib import Path

import torch

from diagnose_cotracker_future_intent_cv import (
    evaluate_task,
)


ROOT = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")


def transfer_target_semantics(
    reliable_motion: torch.Tensor,
    semantic_motion: torch.Tensor,
    semantic_attention: torch.Tensor,
    task: str,
) -> torch.Tensor:
    point_xy = reliable_motion[..., :2]
    seed_xy = semantic_motion[..., :2]
    squared_distance = (point_xy[:, None, :, None] - seed_xy[:, None, None]).square().sum(-1)
    kernel = torch.exp(-squared_distance / (2.0 * 0.22**2))
    semantic_weight = torch.einsum("nls,nlks->nlk", semantic_attention, kernel)

    visibility = reliable_motion[..., 11].clamp(0, 1)
    confidence = reliable_motion[..., 12].clamp(0, 1)
    speed = reliable_motion[..., 8].clamp_min(0)
    speed_scale = speed.quantile(0.75, dim=-1, keepdim=True).clamp_min(1e-4)
    dynamic = (speed / speed_scale).clamp(0, 2) / 2
    support = semantic_weight * visibility[:, None] * confidence[:, None].sqrt()
    support = support * (0.25 + 0.75 * dynamic[:, None])

    x = reliable_motion[..., 0]
    y = reliable_motion[..., 1]
    vx = reliable_motion[..., 2]
    vy = reliable_motion[..., 3]
    result = []
    for target_index in range(semantic_attention.shape[1]):
        target_features = []
        if task == "action":
            centre = (0.0, 0.0, -0.55, 0.55)[target_index]
            width = (0.28, 0.75, 0.30, 0.30)[target_index]
        else:
            centre, width = 0.0, 1.25
        for horizon in (0.0, 2.0, 4.0, 8.0):
            future_x = x + horizon * vx
            future_y = y + horizon * vy
            corridor = torch.exp(-((future_x - centre) / width).square())
            front = torch.sigmoid(3.0 * future_y)
            weight = support[:, target_index] * corridor * front
            normalizer = weight.sum(-1, keepdim=True).clamp_min(1e-5)
            pooled = torch.einsum("nk,nkf->nf", weight / normalizer, reliable_motion)
            top_values, top_indices = weight.topk(k=4, dim=-1)
            top_motion = reliable_motion.gather(
                1, top_indices[..., None].expand(-1, -1, reliable_motion.shape[-1])
            ).flatten(1)
            target_features.extend(
                (
                    pooled,
                    top_motion,
                    top_values,
                    weight.sum(-1, keepdim=True),
                    weight.amax(-1, keepdim=True),
                )
            )
        result.append(torch.cat(target_features, dim=-1))
    return torch.stack(result, dim=1)


def main() -> None:
    reliable_motion = torch.load(ROOT / "cotracker_motion_features_test.pt", map_location="cpu").float()
    semantic_motion = torch.load(ROOT / "relational_motion_features_test.pt", map_location="cpu").float()
    action_attention = torch.load(ROOT / "relational_action_attention_test.pt", map_location="cpu").float()
    reason_attention = torch.load(ROOT / "relational_reason_attention_test.pt", map_location="cpu").float()
    action_features = transfer_target_semantics(
        reliable_motion, semantic_motion, action_attention, "action"
    )
    reason_features = transfer_target_semantics(
        reliable_motion, semantic_motion, reason_attention, "reason"
    )

    calibration = torch.tensor(json.loads((ROOT / "calibration.json").read_text())["image"])
    action_base = torch.load(ROOT / "pre_relational_action_test.pt", map_location="cpu").float()
    action_target = torch.load(ROOT / "action_target_test.pt", map_location="cpu").float()
    reason_base = torch.load(ROOT / "pre_relational_reason_test.pt", map_location="cpu").float()
    reason_target = torch.load(ROOT / "reason_target_test.pt", map_location="cpu").float()
    folds = torch.randperm(len(action_target), generator=torch.Generator().manual_seed(20260825)) % 5
    output = {
        "motion_source": "CoTracker3_offline_grid8",
        "semantic_source": "V8.2_DINO_target_attention_spatial_transport",
        "action": evaluate_task(
            "action", action_base, action_target, calibration[:4], action_features, folds, 0.20
        ),
        "reason": evaluate_task(
            "reason", reason_base, reason_target, calibration[4:], reason_features, folds, 0.16
        ),
    }
    output_path = ROOT / "cotracker_semantic_transport_cv.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
