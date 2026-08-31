from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.evaluate_tida_oia import aie_branch_metrics
from fate_oia.models.vetra_tta_combo_calibrator import VetraTTAComboCalibrator


@dataclass(frozen=True)
class StageCDeployment:
    action_calibrator: VetraTTAComboCalibrator
    reason_thresholds: torch.Tensor
    source_checkpoint_sha256: str


def _required(payload: dict[str, Any], keys: tuple[str, ...], path: Path) -> None:
    missing = sorted(set(keys).difference(payload))
    if missing:
        raise RuntimeError(f"Stage-C artifact {path} is missing keys: {missing}")


def load_stage_c_deployment(path: str | Path, device: torch.device) -> StageCDeployment:
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = (
        "mean", "scale", "coefficient", "intercept", "class_codes",
        "action_thresholds", "reason_thresholds", "original_weight",
        "source_checkpoint_sha256",
    )
    _required(payload, keys, path)
    calibrator = VetraTTAComboCalibrator(
        mean=payload["mean"],
        scale=payload["scale"],
        coefficient=payload["coefficient"],
        intercept=payload["intercept"],
        class_codes=payload["class_codes"],
        thresholds=payload["action_thresholds"],
        original_weight=float(payload["original_weight"]),
    ).to(device)
    calibrator.eval()
    return StageCDeployment(
        action_calibrator=calibrator,
        reason_thresholds=torch.as_tensor(
            payload["reason_thresholds"], dtype=torch.float32, device=device
        ),
        source_checkpoint_sha256=str(payload["source_checkpoint_sha256"]),
    )


def verify_stage_c_source(deployment: StageCDeployment, checkpoint: str | Path) -> None:
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    if digest.lower() != deployment.source_checkpoint_sha256.lower():
        raise RuntimeError(
            "Stage-C deployment source hash does not match the frozen Stage-B checkpoint"
        )


def _reason_deploy_logits(logits: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    thresholds = thresholds.to(logits).clamp(1e-6, 1.0 - 1e-6)
    if thresholds.numel() != logits.shape[-1]:
        raise ValueError("reason threshold count does not match reason logits")
    return logits - torch.logit(thresholds)[None]


@torch.no_grad()
def evaluate_stage_c_branches(
    rows: dict[str, torch.Tensor], deployment: StageCDeployment
) -> dict[str, dict[str, torch.Tensor]]:
    required = (
        "image_action_original", "image_action_flip", "action_original",
        "action_flip", "image_reason_original", "reason_original",
    )
    missing = sorted(set(required).difference(rows))
    if missing:
        raise KeyError(f"TIDA TTA rows are missing: {missing}")
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for branch, action_original, action_flip, reason in (
        (
            "image", rows["image_action_original"], rows["image_action_flip"],
            rows["image_reason_original"],
        ),
        (
            "video", rows["action_original"], rows["action_flip"],
            rows["reason_original"],
        ),
    ):
        action = deployment.action_calibrator(action_original, action_flip)
        outputs[branch] = {
            **action,
            "reason_logits": reason,
            "reason_deploy_logits": _reason_deploy_logits(
                reason, deployment.reason_thresholds
            ),
        }
    return outputs


def _load_rows(root: Path, split: str) -> dict[str, torch.Tensor]:
    split_dir = root / split
    keys = (
        "action_original", "action_flip", "image_action_original",
        "image_action_flip", "reason_original", "image_reason_original",
        "action_target", "reason_target",
    )
    return {
        key: torch.load(split_dir / f"{key}.pt", map_location="cpu", weights_only=True)
        for key in keys
    }


def _metrics(
    output: dict[str, torch.Tensor], rows: dict[str, torch.Tensor]
) -> dict[str, Any]:
    return aie_branch_metrics(
        output["action_deploy_logits"], output["reason_deploy_logits"],
        rows["action_target"], rows["reason_target"], threshold=0.5,
    )


def _numeric_delta(video: dict[str, Any], image: dict[str, Any]) -> dict[str, float]:
    keys = ("Act_mF1", "Act_oF1", "Act_mAP", "Exp_mF1", "Exp_oF1", "Exp_mAP", "joint")
    return {key: float(video[key]) - float(image[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta-output-root", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--image-checkpoint")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    deployment = load_stage_c_deployment(args.deployment, device)
    if args.image_checkpoint:
        verify_stage_c_source(deployment, args.image_checkpoint)
    rows = {
        key: value.to(device)
        for key, value in _load_rows(Path(args.tta_output_root), args.split).items()
    }
    branches = evaluate_stage_c_branches(rows, deployment)
    image_metrics = _metrics(branches["image"], rows)
    video_metrics = _metrics(branches["video"], rows)
    result = {
        "split": args.split,
        "sample_count": int(rows["action_target"].shape[0]),
        "deployment": str(Path(args.deployment).resolve()),
        "source_checkpoint_sha256": deployment.source_checkpoint_sha256,
        "test_labels_used_for_deployment_fit": False,
        "image_stage_c": image_metrics,
        "video_stage_c": video_metrics,
        "video_minus_image": _numeric_delta(video_metrics, image_metrics),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for branch, output in branches.items():
        torch.save(
            output["action_deploy_logits"].cpu(),
            output_dir / f"{branch}_action_stage_c_deploy_logits.pt",
        )
        torch.save(
            output["reason_deploy_logits"].cpu(),
            output_dir / f"{branch}_reason_stage_c_deploy_logits.pt",
        )
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
