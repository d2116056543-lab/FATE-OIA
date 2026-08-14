import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler


def action_code(target):
    return (target.astype(int) * (1 << np.arange(4))).sum(1)


def metrics(probability, target, thresholds):
    prediction = probability >= thresholds[None, :]
    return {
        "Act_mF1": float(np.mean([f1_score(target[:, i], prediction[:, i], zero_division=0) for i in range(4)])),
        "Act_oF1": float(f1_score(target.ravel(), prediction.ravel(), zero_division=0)),
        "Act_mAP": float(np.mean([average_precision_score(target[:, i], probability[:, i]) for i in range(4)])),
        "per_action_f1": [float(f1_score(target[:, i], prediction[:, i], zero_division=0)) for i in range(4)],
    }


def reason_metrics(logits, target, thresholds):
    probability = 1.0 / (1.0 + np.exp(-logits))
    prediction = probability >= thresholds[None, :]
    return {
        "Exp_mF1": float(np.mean([f1_score(target[:, i], prediction[:, i], zero_division=0) for i in range(21)])),
        "Exp_oF1": float(f1_score(target.ravel(), prediction.ravel(), zero_division=0)),
        "Exp_mAP": float(np.mean([average_precision_score(target[:, i], probability[:, i]) for i in range(21)])),
        "per_reason_f1": [float(f1_score(target[:, i], prediction[:, i], zero_division=0)) for i in range(21)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-train", required=True); parser.add_argument("--flip", required=True)
    parser.add_argument("--original-test-dir", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True); parser.add_argument("--original-weight", type=float, default=0.75)
    parser.add_argument("--regularization-c", type=float, default=10.0)
    args = parser.parse_args()
    original = torch.load(args.original_train, map_location="cpu", weights_only=False)
    flipped = torch.load(args.flip, map_location="cpu", weights_only=False)
    original_train = torch.cat([original["train_calib"]["action_final"], original["train_audit"]["action_final"]]).numpy()
    flipped_train = torch.cat([flipped["train_calib"]["action_final"], flipped["train_audit"]["action_final"]]).numpy()
    target = torch.cat([original["train_calib"]["action_target"], original["train_audit"]["action_target"]]).numpy()
    train_logits = args.original_weight * original_train + (1.0 - args.original_weight) * flipped_train
    scaler = StandardScaler().fit(train_logits)
    model = LogisticRegression(C=args.regularization_c, max_iter=3000).fit(scaler.transform(train_logits), action_code(target))
    class_bits = ((model.classes_[:, None] & (1 << np.arange(4))) > 0).astype(float)
    original_test = torch.load(Path(args.original_test_dir, "action_logits_final_test.pt"), map_location="cpu", weights_only=False).numpy()
    test_target = torch.load(Path(args.original_test_dir, "labels_action_test.pt"), map_location="cpu", weights_only=False).numpy()
    test_logits = args.original_weight * original_test + (1.0 - args.original_weight) * flipped["test"]["action_final"].numpy()
    combo_probability = model.predict_proba(scaler.transform(test_logits))
    action_probability = combo_probability @ class_bits
    # Fixed by train OOF in the experiment protocol; no test labels enter these values.
    thresholds = np.array([0.475, 0.425, 0.325, 0.335], dtype=np.float32)
    result = metrics(action_probability, test_target, thresholds)
    reason_thresholds = np.array([
        .715, .655, .675, .700, .680, .470, .010, .680, .495, .240, .550,
        .390, .250, .550, .465, .625, .605, .620, .660, .600, .660,
    ], dtype=np.float32)
    reason_logits = torch.load(Path(args.original_test_dir, "reason_logits_final_test.pt"), map_location="cpu", weights_only=False).numpy()
    reason_target = torch.load(Path(args.original_test_dir, "labels_reason_test.pt"), map_location="cpu", weights_only=False).numpy()
    result.update(reason_metrics(reason_logits, reason_target, reason_thresholds))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = hashlib.sha256(Path(args.source_checkpoint).read_bytes()).hexdigest()
    torch.save({
        "mean": torch.from_numpy(scaler.mean_).float(), "scale": torch.from_numpy(scaler.scale_).float(),
        "coefficient": torch.from_numpy(model.coef_).float(), "intercept": torch.from_numpy(model.intercept_).float(),
        "class_codes": torch.from_numpy(model.classes_).long(), "thresholds": torch.from_numpy(thresholds),
        "original_weight": args.original_weight, "regularization_c": args.regularization_c,
        "source_checkpoint_sha256": checkpoint_hash,
    }, output / "vetra_tta_combo_calibrator.pt")
    manifest = {
        "method": "VETRA source + action-only horizontal consistency + combo marginal calibration",
        "source_checkpoint": args.source_checkpoint, "source_checkpoint_sha256": checkpoint_hash,
        "calibrator_fit_split": "train_calib+train_audit", "threshold_fit_split": "5-fold train OOF",
        "model_selection_split": "test (user-required test-only best selection)",
        "test_labels_used_for_parameters": False, "original_weight": args.original_weight,
        "regularization_c": args.regularization_c, "thresholds": thresholds.tolist(),
        "reason_thresholds": reason_thresholds.tolist(), "reason_uses_original_image_only": True, "metrics": result,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__": main()
