from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.tida_artifacts import atomic_write_json, file_sha256, save_checkpoint_atomic
from fate_oia.utils.tida_contracts import select_reason_beta
from fate_oia.utils.vetra_from_scratch import (
    fit_action_combo_oof,
    fit_label_thresholds,
    select_action_combo_hyperparameters,
)


TENSOR_KEYS = (
    "action_original", "action_flip", "image_action_original", "image_action_flip",
    "reason_original", "image_reason_original", "action_target", "reason_target",
)


def _load_split(root: Path, split: str) -> dict[str, Any]:
    directory = root / split
    rows = {key: torch.load(directory / f"{key}.pt", map_location="cpu", weights_only=True) for key in TENSOR_KEYS}
    rows["file_names"] = json.loads((directory / "file_names.json").read_text(encoding="utf-8"))
    rows["concept_path"] = directory / "dynamic_explanation_examples.jsonl"
    return rows


def _membership(classes: np.ndarray, action_dim: int = 4) -> np.ndarray:
    return ((classes[:, None] & (1 << np.arange(action_dim))) > 0).astype(np.float64)


def _combo_probability(fitted: dict[str, Any], logits: np.ndarray) -> np.ndarray:
    scaler, model = fitted["scaler"], fitted["model"]
    return model.predict_proba(scaler.transform(logits)) @ _membership(model.classes_)


def fit_deployment_parameters(calib: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    original = torch.cat([calib["action_original"], audit["action_original"]]).numpy()
    flipped = torch.cat([calib["action_flip"], audit["action_flip"]]).numpy()
    action_target = torch.cat([calib["action_target"], audit["action_target"]]).numpy()
    selection = select_action_combo_hyperparameters(
        original, flipped, action_target,
        original_weights=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
        regularization_cs=(0.1, 1.0, 10.0, 100.0), outer_folds=5, inner_folds=4, seed=20260821,
    )
    weight = float(selection["selected_original_weight"])
    regularization = float(selection["selected_regularization_c"])
    mixed = weight * original + (1.0 - weight) * flipped
    combo = fit_action_combo_oof(mixed, action_target, regularization_c=regularization, folds=5, seed=20260821)
    action_threshold = fit_label_thresholds(combo["oof_action_probability"], action_target)
    reason = select_reason_beta(
        calib["image_reason_original"], calib["reason_original"] - calib["image_reason_original"], calib["reason_target"],
        audit["image_reason_original"], audit["reason_original"] - audit["image_reason_original"], audit["reason_target"],
        candidates=(0.0, 0.25, 0.5, 0.75, 1.0), folds=5,
    )
    return {
        "selection": selection, "action_original_weight": weight, "action_regularization_c": regularization,
        "combo": combo, "action_threshold": torch.from_numpy(action_threshold),
        "reason_beta": reason["reason_beta"].cpu(), "reason_threshold": reason["reason_threshold"].cpu(),
        "reason_oof_scores": reason["oof_scores"].cpu(),
    }


def apply_deployment(parameters: dict[str, Any], rows: dict[str, Any]) -> dict[str, torch.Tensor]:
    weight = parameters["action_original_weight"]
    mixed = weight * rows["action_original"].numpy() + (1.0 - weight) * rows["action_flip"].numpy()
    action_probability = torch.from_numpy(_combo_probability(parameters["combo"], mixed)).float().clamp(1e-6, 1 - 1e-6)
    action_logits = torch.logit(action_probability)
    reason_logits = rows["image_reason_original"] + parameters["reason_beta"][None] * (
        rows["reason_original"] - rows["image_reason_original"]
    )
    return {"action_logits": action_logits, "reason_logits": reason_logits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tta-output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    tta_root, output_dir = Path(args.tta_output_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calib, audit, test = (_load_split(tta_root, split) for split in ("train_calib", "train_audit", "test"))
    parameters = fit_deployment_parameters(calib, audit)
    deployed = apply_deployment(parameters, test)
    thresholds = torch.cat([parameters["action_threshold"], parameters["reason_threshold"]])
    metrics = aie_branch_metrics(
        deployed["action_logits"], deployed["reason_logits"], test["action_target"], test["reason_target"], thresholds
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    deploy_payload = {
        "tida_trainable_state": checkpoint["tida_trainable_state"],
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "action_original_weight": parameters["action_original_weight"],
        "action_regularization_c": parameters["action_regularization_c"],
        "action_combo_scaler": parameters["combo"]["scaler"],
        "action_combo_model": parameters["combo"]["model"],
        "action_threshold": parameters["action_threshold"],
        "reason_beta": parameters["reason_beta"], "reason_threshold": parameters["reason_threshold"],
        "test_labels_used_for_parameter_fit": False,
    }
    deploy_path = output_dir / "tida_oia_v1_deploy.pth"
    save_checkpoint_atomic(deploy_path, deploy_payload)
    manifest = {
        "pass": True, "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "deployment_sha256": file_sha256(deploy_path), "fit_splits": ["train_calib", "train_audit"],
        "evaluation_split": "test", "test_labels_used_for_parameter_fit": False,
        "action_original_weight": parameters["action_original_weight"],
        "action_regularization_c": parameters["action_regularization_c"],
        "reason_tta": "original_only", "reason_beta": parameters["reason_beta"].tolist(),
        "thresholds": thresholds.tolist(), "nested_oof_action_selection": parameters["selection"],
    }
    atomic_write_json(output_dir / "tida_oia_v1_deployment_manifest.json", manifest)
    atomic_write_json(output_dir / "metrics_summary.json", metrics)
    atomic_write_json(output_dir / "per_label_metrics.json", {
        key: value for key, value in metrics.items() if "per_label" in key
    })
    shutil.copyfile(test["concept_path"], output_dir / "dynamic_explanation_examples.jsonl")
    torch.save(deployed["action_logits"], output_dir / "action_logits_deploy_test.pt")
    torch.save(deployed["reason_logits"], output_dir / "reason_logits_deploy_test.pt")
    print(json.dumps({"event": "tida_deployment", **metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
