from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds


@dataclass(frozen=True)
class DualSnapshotWeights:
    action_late: float = 0.65
    reason_late: float = 0.875

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


def blend_snapshots(early: Tensor, late: Tensor, late_weight: float) -> Tensor:
    if early.shape != late.shape:
        raise ValueError(f"Snapshot shapes differ: {tuple(early.shape)} != {tuple(late.shape)}")
    return float(late_weight) * late.float() + (1.0 - float(late_weight)) * early.float()


def _fit_thresholds(logits: Tensor, labels: Tensor, shrinkage: float) -> Tensor:
    return fit_posthoc_thresholds(
        logits.float(),
        labels.float(),
        [list(range(logits.shape[1]))],
        shrinkage_support=float(shrinkage),
        grid_step=0.01,
    )["threshold_prob"]


def fit_dual_thresholds(
    early_action: Tensor,
    late_action: Tensor,
    early_reason: Tensor,
    late_reason: Tensor,
    action_target: Tensor,
    reason_target: Tensor,
    weights: DualSnapshotWeights,
    *,
    action_shrinkage: float = 50.0,
    reason_shrinkage: float = 0.0,
) -> dict[str, Any]:
    action_logits = blend_snapshots(early_action, late_action, weights.action_late)
    reason_logits = blend_snapshots(early_reason, late_reason, weights.reason_late)
    if action_logits.shape != action_target.shape or reason_logits.shape != reason_target.shape:
        raise ValueError("Calibration logits and labels must have identical shapes")
    return {
        "action_thresholds": _fit_thresholds(action_logits, action_target, action_shrinkage),
        "reason_thresholds": _fit_thresholds(reason_logits, reason_target, reason_shrinkage),
        "action_shrinkage": float(action_shrinkage),
        "reason_shrinkage": float(reason_shrinkage),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_epoch(path: Path) -> dict[str, Any]:
    calibration = torch.load(path / "train_calib_logits.pt", map_location="cpu", weights_only=True)
    return {
        "calibration": calibration,
        "calibration_names": _read_json(path / "file_names_train_calib.json"),
        "test_names": _read_json(path / "file_names_test.json"),
        "action_test": torch.load(path / "action_logits_final_test.pt", map_location="cpu", weights_only=True),
        "reason_test": torch.load(path / "reason_logits_final_test.pt", map_location="cpu", weights_only=True),
        "action_test_target": torch.load(path / "labels_action_test.pt", map_location="cpu", weights_only=True),
        "reason_test_target": torch.load(path / "labels_reason_test.pt", map_location="cpu", weights_only=True),
    }


def load_snapshot_artifacts(early_dir: str | Path, late_dir: str | Path) -> dict[str, Any]:
    early, late = _load_epoch(Path(early_dir)), _load_epoch(Path(late_dir))
    if early["calibration_names"] != late["calibration_names"]:
        raise ValueError("Snapshot calibration file names are not identically aligned")
    if early["test_names"] != late["test_names"]:
        raise ValueError("Snapshot test file names are not identically aligned")
    for key in ("action_labels", "reason_labels"):
        if not torch.equal(early["calibration"][key], late["calibration"][key]):
            raise ValueError(f"Snapshot calibration {key} differ")
    for key in ("action_test_target", "reason_test_target"):
        if not torch.equal(early[key], late[key]):
            raise ValueError(f"Snapshot {key} differ")
    return {"early": early, "late": late}


def _metrics(action_logits: Tensor, reason_logits: Tensor, action_target: Tensor, reason_target: Tensor) -> dict[str, Any]:
    action = multilabel_metrics_from_logits(action_logits.float(), action_target.float(), prefix="Act_")
    reason = multilabel_metrics_from_logits(reason_logits.float(), reason_target.float(), prefix="Exp_")
    return {**action, **reason, "joint": 0.5 * (float(action["Act_mF1"]) + float(reason["Exp_mF1"]))}


def _bootstrap(candidate: Tensor, baseline: Tensor, target: Tensor, prefix: str, samples: int, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    differences = []
    metric = f"{prefix}mF1"
    for _ in range(samples):
        index = torch.randint(0, target.shape[0], (target.shape[0],), generator=generator)
        candidate_score = float(multilabel_metrics_from_logits(candidate[index], target[index], prefix=prefix)[metric])
        baseline_score = float(multilabel_metrics_from_logits(baseline[index], target[index], prefix=prefix)[metric])
        differences.append(candidate_score - baseline_score)
    values = torch.tensor(differences)
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.quantile(0.025)),
        "ci95_high": float(values.quantile(0.975)),
        "probability_positive": float((values > 0).float().mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_dual_snapshot(
    early_dir: str | Path,
    late_dir: str | Path,
    weights: DualSnapshotWeights,
    *,
    action_shrinkage: float = 50.0,
    reason_shrinkage: float = 0.0,
    bootstrap_samples: int = 1000,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    data = load_snapshot_artifacts(early_dir, late_dir)
    early, late = data["early"], data["late"]
    thresholds = fit_dual_thresholds(
        early["calibration"]["action_logits"], late["calibration"]["action_logits"],
        early["calibration"]["reason_logits"], late["calibration"]["reason_logits"],
        early["calibration"]["action_labels"], early["calibration"]["reason_labels"],
        weights, action_shrinkage=action_shrinkage, reason_shrinkage=reason_shrinkage,
    )
    action = blend_snapshots(early["action_test"], late["action_test"], weights.action_late)
    reason = blend_snapshots(early["reason_test"], late["reason_test"], weights.reason_late)
    deploy_action = apply_posthoc_threshold(action, thresholds["action_thresholds"])
    deploy_reason = apply_posthoc_threshold(reason, thresholds["reason_thresholds"])
    late_action_threshold = _fit_thresholds(late["calibration"]["action_logits"], late["calibration"]["action_labels"], action_shrinkage)
    late_reason_threshold = _fit_thresholds(late["calibration"]["reason_logits"], late["calibration"]["reason_labels"], reason_shrinkage)
    late_deploy_action = apply_posthoc_threshold(late["action_test"], late_action_threshold)
    late_deploy_reason = apply_posthoc_threshold(late["reason_test"], late_reason_threshold)
    action_target, reason_target = early["action_test_target"], early["reason_test_target"]
    result = {
        "contract": {
            "weights": asdict(weights),
            "action_shrinkage": action_shrinkage,
            "reason_shrinkage": reason_shrinkage,
            "selection_source": "locked_from_prior_train_calib_diagnostics_before_new_test",
            "test_parameter_writeback": False,
        },
        "raw": _metrics(action, reason, action_target, reason_target),
        "deploy": _metrics(deploy_action, deploy_reason, action_target, reason_target),
        "late_snapshot_deploy": _metrics(late_deploy_action, late_deploy_reason, action_target, reason_target),
        "thresholds": {
            "action": thresholds["action_thresholds"].tolist(),
            "reason": thresholds["reason_thresholds"].tolist(),
        },
        "bootstrap_vs_late": {
            "action": _bootstrap(deploy_action, late_deploy_action, action_target, "Act_", bootstrap_samples, 20260810),
            "reason": _bootstrap(deploy_reason, late_deploy_reason, reason_target, "Exp_", bootstrap_samples, 20260811),
        },
    }
    tensors = {
        "action_logits": action,
        "reason_logits": reason,
        "action_deploy_logits": deploy_action,
        "reason_deploy_logits": deploy_reason,
        "action_labels": action_target,
        "reason_labels": reason_target,
    }
    return result, tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-dir", required=True)
    parser.add_argument("--late-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-late-weight", type=float, default=0.65)
    parser.add_argument("--reason-late-weight", type=float, default=0.875)
    parser.add_argument("--action-shrinkage", type=float, default=50.0)
    parser.add_argument("--reason-shrinkage", type=float, default=0.0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result, tensors = evaluate_dual_snapshot(
        args.early_dir,
        args.late_dir,
        DualSnapshotWeights(args.action_late_weight, args.reason_late_weight),
        action_shrinkage=args.action_shrinkage,
        reason_shrinkage=args.reason_shrinkage,
        bootstrap_samples=args.bootstrap_samples,
    )
    early_dir, late_dir = Path(args.early_dir), Path(args.late_dir)
    result["artifact_hashes"] = {
        "early_action_logits": _sha256(early_dir / "action_logits_final_test.pt"),
        "early_reason_logits": _sha256(early_dir / "reason_logits_final_test.pt"),
        "late_action_logits": _sha256(late_dir / "action_logits_final_test.pt"),
        "late_reason_logits": _sha256(late_dir / "reason_logits_final_test.pt"),
    }
    (output / "dual_snapshot_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "deployment_contract.json").write_text(json.dumps(result["contract"], indent=2), encoding="utf-8")
    torch.save(tensors, output / "dual_snapshot_test_tensors.pt")
    print(json.dumps({"event": "dual_snapshot_result", **result["deploy"]}), flush=True)


if __name__ == "__main__":
    main()
