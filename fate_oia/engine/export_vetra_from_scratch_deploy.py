from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from fate_oia.utils.vetra_from_scratch import fit_action_combo_oof, fit_label_thresholds


def multilabel_metrics(prefix: str, probability: np.ndarray, target: np.ndarray, thresholds: np.ndarray):
    prediction = probability >= thresholds[None, :]
    return {
        f"{prefix}_mF1": float(np.mean([f1_score(target[:, i], prediction[:, i], zero_division=0) for i in range(target.shape[1])])),
        f"{prefix}_oF1": float(f1_score(target.ravel(), prediction.ravel(), zero_division=0)),
        f"{prefix}_mAP": float(np.mean([average_precision_score(target[:, i], probability[:, i]) for i in range(target.shape[1])])),
        f"{prefix}_per_label_f1": [float(f1_score(target[:, i], prediction[:, i], zero_division=0)) for i in range(target.shape[1])],
    }


def concatenate(payload, view: str, key: str, splits: tuple[str, ...]) -> np.ndarray:
    return torch.cat([payload[view][split][key] for split in splits]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--original-weight", type=float, default=0.75)
    parser.add_argument("--regularization-c", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--fit-splits", nargs="+", default=["train_calib"])
    args = parser.parse_args()

    payload = torch.load(args.outputs, map_location="cpu", weights_only=False)
    fit_splits = tuple(args.fit_splits)
    if any(split == "test" for split in fit_splits):
        raise RuntimeError("test outputs cannot be used to fit deployment parameters")
    original_action = concatenate(payload, "original", "action_final", fit_splits)
    flipped_action = concatenate(payload, "flip", "action_final", fit_splits)
    action_target = concatenate(payload, "original", "action_target", fit_splits)
    mixed_action = args.original_weight * original_action + (1.0 - args.original_weight) * flipped_action
    fitted = fit_action_combo_oof(
        mixed_action,
        action_target,
        regularization_c=args.regularization_c,
        folds=args.folds,
        seed=args.seed,
    )
    action_thresholds = fit_label_thresholds(fitted["oof_action_probability"], action_target)
    scaler, model = fitted["scaler"], fitted["model"]
    class_bits = ((model.classes_[:, None] & (1 << np.arange(action_target.shape[1]))) > 0).astype(float)

    original_test = payload["original"]["test"]["action_final"].numpy()
    flipped_test = payload["flip"]["test"]["action_final"].numpy()
    mixed_test = args.original_weight * original_test + (1.0 - args.original_weight) * flipped_test
    action_probability = model.predict_proba(scaler.transform(mixed_test)) @ class_bits
    test_action_target = payload["original"]["test"]["action_target"].numpy()

    reason_train_logits = concatenate(payload, "original", "reason_final", fit_splits)
    reason_train_target = concatenate(payload, "original", "reason_target", fit_splits)
    reason_train_probability = 1.0 / (1.0 + np.exp(-reason_train_logits))
    reason_thresholds = fit_label_thresholds(reason_train_probability, reason_train_target)
    reason_test_logits = payload["original"]["test"]["reason_final"].numpy()
    reason_probability = 1.0 / (1.0 + np.exp(-reason_test_logits))
    test_reason_target = payload["original"]["test"]["reason_target"].numpy()

    result = {
        **multilabel_metrics("Act", action_probability, test_action_target, action_thresholds),
        **multilabel_metrics("Exp", reason_probability, test_reason_target, reason_thresholds),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(Path(args.source_checkpoint).read_bytes()).hexdigest()
    torch.save(
        {
            "mean": torch.from_numpy(scaler.mean_).float(),
            "scale": torch.from_numpy(scaler.scale_).float(),
            "coefficient": torch.from_numpy(model.coef_).float(),
            "intercept": torch.from_numpy(model.intercept_).float(),
            "class_codes": torch.from_numpy(model.classes_).long(),
            "action_thresholds": torch.from_numpy(action_thresholds),
            "reason_thresholds": torch.from_numpy(reason_thresholds),
            "original_weight": args.original_weight,
            "source_checkpoint_sha256": checkpoint_hash,
        },
        output / "vetra_from_scratch_deploy.pth",
    )
    manifest = {
        "method": "clean AIE training + internal low-LR stage + action flip/combo deployment",
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "source_checkpoint_sha256": checkpoint_hash,
        "action_calibrator_fit_split": "+".join(fit_splits),
        "action_threshold_fit_split": f"{args.folds}-fold train OOF",
        "reason_threshold_fit_split": "+".join(fit_splits),
        "test_labels_used_for_parameters": False,
        "original_weight": args.original_weight,
        "regularization_c": args.regularization_c,
        "action_thresholds": action_thresholds.tolist(),
        "reason_thresholds": reason_thresholds.tolist(),
        "metrics": result,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
