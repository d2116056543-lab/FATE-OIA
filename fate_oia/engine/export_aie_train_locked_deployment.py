from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from fate_oia.engine.train_aie_oia import (
    build_model,
    canonical_model_state_dict,
    compatible_checkpoint_state_dict,
    load_config,
)
from fate_oia.models.aie_train_locked_deployment import AIETrainLockedDeployment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--diagnostic-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_path = Path(args.source_checkpoint).resolve()
    diagnostic_path = Path(args.diagnostic_json).resolve()
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    protocol = str(diagnostic.get("primary_protocol", ""))
    if protocol != "five_fold_train_only_threshold_median":
        raise RuntimeError(f"refusing non-train-locked threshold protocol: {protocol!r}")

    action_scales = diagnostic["action_scales"]
    threshold_prob = diagnostic["median_threshold"]
    config = load_config(args.config)
    model = build_model(config, torch.device("cpu"))
    source = torch.load(source_path, map_location="cpu")
    model.load_state_dict(
        compatible_checkpoint_state_dict(model, source.get("model", source)), strict=True
    )
    reason_action_scale = float(diagnostic.get("reason_action_scale", 0.0))
    reason_config = config["reason_private"]
    reason_scale = float(
        diagnostic.get(
            "reason_scale",
            reason_config.get("reason_scale", reason_config["reason_scale_max"]),
        )
    )
    deployment = AIETrainLockedDeployment(
        model,
        action_scales,
        threshold_prob,
        reason_action_scale=reason_action_scale,
        reason_scale=reason_scale,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": canonical_model_state_dict(model.state_dict()),
        "deployment": dict(deployment.deployment_state()),
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256(source_path),
        "diagnostic_json": str(diagnostic_path),
        "diagnostic_json_sha256": sha256(diagnostic_path),
        "selection_protocol": protocol,
        "parameter_fit_split": "train_calib/train_audit only",
        "test_labels_used_for_parameters": False,
        "reason_action_scale": reason_action_scale,
        "reason_scale": reason_scale,
    }
    torch.save(payload, output)
    manifest = {key: value for key, value in payload.items() if key not in {"model", "deployment"}}
    manifest.update(
        {
            "action_scales": list(map(float, action_scales)),
            "threshold_prob": list(map(float, threshold_prob)),
            "output": str(output.resolve()),
            "output_sha256": sha256(output),
        }
    )
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
