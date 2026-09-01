from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load(root: Path, name: str) -> torch.Tensor:
    return torch.load(root / f"{name}_test.pt", map_location="cpu", weights_only=True).float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", required=True)
    args = parser.parse_args()
    root = Path(args.epoch_dir)
    candidate = load(root, "reason_local_candidate_delta")
    shuffled = load(root, "reason_local_shuffled_delta")
    selected = load(root, "reason_local_selected_deleted_delta")
    control = load(root, "reason_local_random_deleted_delta")
    gap = load(root, "reason_local_selected_minus_random_gap")
    deploy = load(root, "reason_local_deploy_delta")
    gate = load(root, "reason_local_deploy_gate")
    velocity = load(root, "reason_local_velocity_rms")
    acceleration = load(root, "reason_local_acceleration_rms")
    payload = {
        "candidate_rms": float(candidate.square().mean().sqrt()),
        "candidate_nonzero_rate": float((candidate.abs() > 1e-6).float().mean()),
        "order_delta_gap_mean": float((candidate - shuffled).abs().mean()),
        "selected_deletion_effect_mean": float((candidate - selected).abs().mean()),
        "control_deletion_effect_mean": float((candidate - control).abs().mean()),
        "selected_minus_control_gap_mean": float(gap.mean()),
        "selected_minus_control_positive_rate": float((gap > 0).float().mean()),
        "deploy_delta_rms": float(deploy.square().mean().sqrt()),
        "deploy_open_rate": float((gate > 0).float().mean()),
        "velocity_rms_mean": float(velocity.mean()),
        "acceleration_rms_mean": float(acceleration.mean()),
        "per_reason_candidate_rms": candidate.square().mean(0).sqrt().tolist(),
        "per_reason_deletion_gap": gap.mean(0).tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
