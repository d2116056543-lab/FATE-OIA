from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compose_deployment_parameters(
    *,
    action_scale_raw: Tensor,
    decision_threshold_raw: Tensor,
    locked_threshold_prob: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compose trained action decisions with the original train-locked reason boundary."""
    action_scale_raw = torch.as_tensor(action_scale_raw, dtype=torch.float32).view(-1)
    threshold_raw = torch.as_tensor(decision_threshold_raw, dtype=torch.float32).view(-1)
    locked = torch.as_tensor(locked_threshold_prob, dtype=torch.float32).view(-1)
    if action_scale_raw.numel() != 4 or threshold_raw.numel() != 25 or locked.numel() != 25:
        raise ValueError("expected 4 action scales and two 25-label threshold vectors")
    action_scales = torch.sigmoid(action_scale_raw)
    action_thresholds = 0.05 + 0.90 * torch.sigmoid(threshold_raw[:4])
    thresholds = torch.cat((action_thresholds, locked[4:].clone()))
    if not bool(torch.isfinite(action_scales).all() and torch.isfinite(thresholds).all()):
        raise ValueError("deployment parameters must be finite")
    if bool(((thresholds <= 0.0) | (thresholds >= 1.0)).any()):
        raise ValueError("deployment thresholds must lie strictly between zero and one")
    return action_scales, thresholds


def compose_payload(source: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    if bool(source.get("test_labels_used_for_parameters", True)):
        raise RuntimeError("source deployment is not certified train-only")
    source_sha = str(source.get("source_checkpoint_sha256", ""))
    decision_sha = str(decision.get("source_checkpoint_sha256", ""))
    if not source_sha or source_sha != decision_sha:
        raise RuntimeError("source and decision checkpoints do not share the exact model checkpoint")
    state = decision["decision_state"]
    action_scales, thresholds = compose_deployment_parameters(
        action_scale_raw=state["action_scale_raw"],
        decision_threshold_raw=state["threshold_raw"],
        locked_threshold_prob=source["deployment"]["threshold_prob"],
    )
    deployment = dict(source["deployment"])
    deployment["action_scales"] = action_scales
    deployment["threshold_prob"] = thresholds
    return {
        "model": source["model"],
        "deployment": deployment,
        "source_checkpoint": source["source_checkpoint"],
        "source_checkpoint_sha256": source_sha,
        "selection_protocol": "gradient_trained_action_plus_five_fold_train_locked_reason",
        "parameter_fit_split": "train_calib/train_audit only",
        "test_labels_used_for_parameters": False,
        "provenance": {
            "action_scales": "gradient-trained decision checkpoint",
            "action_thresholds": "gradient-trained decision checkpoint",
            "reason_thresholds": "original five-fold train-only locked deployment",
            "reason_model": "unchanged source checkpoint",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-deployment", required=True)
    parser.add_argument("--decision-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_deployment).resolve()
    decision_path = Path(args.decision_checkpoint).resolve()
    output_path = Path(args.output).resolve()
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    decision = torch.load(decision_path, map_location="cpu", weights_only=False)
    payload = compose_payload(source, decision)
    payload["source_deployment"] = str(source_path)
    payload["source_deployment_sha256"] = sha256(source_path)
    payload["decision_checkpoint"] = str(decision_path)
    payload["decision_checkpoint_sha256"] = sha256(decision_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    manifest = {key: value for key, value in payload.items() if key not in {"model", "deployment"}}
    manifest["action_scales"] = payload["deployment"]["action_scales"].tolist()
    manifest["threshold_prob"] = payload["deployment"]["threshold_prob"].tolist()
    manifest["output"] = str(output_path)
    manifest["output_sha256"] = sha256(output_path)
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
