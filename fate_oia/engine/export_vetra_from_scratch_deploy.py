from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score

from fate_oia.utils.vetra_from_scratch import (
    fit_action_combo_oof,
    fit_label_thresholds,
    fit_prior_anchored_label_thresholds,
    fit_stable_label_thresholds,
    select_action_combo_hyperparameters,
)


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


def resolve_fit_splits(action_splits, reason_splits):
    action = tuple(action_splits)
    reason = action if reason_splits is None else tuple(reason_splits)
    for name, splits in (("action", action), ("reason", reason)):
        if not splits:
            raise ValueError(f"{name} fit splits must be non-empty")
        if "test" in splits:
            raise ValueError(f"test outputs cannot be used to fit {name} deployment parameters")
    return action, reason


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
    parser.add_argument("--reason-fit-splits", nargs="+", default=None)
    parser.add_argument("--select-hyperparameters", action="store_true")
    parser.add_argument(
        "--candidate-original-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
    )
    parser.add_argument(
        "--candidate-regularization-cs",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument("--selection-outer-folds", type=int, default=5)
    parser.add_argument("--selection-inner-folds", type=int, default=4)
    parser.add_argument("--stable-action-thresholds", action="store_true")
    parser.add_argument("--threshold-folds", type=int, default=10)
    parser.add_argument(
        "--reason-threshold-mode",
        choices=("independent", "prior_anchored_train_oof"),
        default="independent",
    )
    parser.add_argument("--reason-threshold-prior", nargs="+", type=float)
    parser.add_argument("--reason-prior-min-macro-gain", type=float, default=0.001)
    parser.add_argument("--reason-prior-alpha-step", type=float, default=0.05)
    parser.add_argument("--reason-threshold-folds", type=int, default=5)
    args = parser.parse_args()

    payload = torch.load(args.outputs, map_location="cpu", weights_only=False)
    fit_splits, reason_fit_splits = resolve_fit_splits(
        args.fit_splits, args.reason_fit_splits
    )
    original_action = concatenate(payload, "original", "action_final", fit_splits)
    flipped_action = concatenate(payload, "flip", "action_final", fit_splits)
    action_target = concatenate(payload, "original", "action_target", fit_splits)
    hyperparameter_selection = None
    if args.select_hyperparameters:
        hyperparameter_selection = select_action_combo_hyperparameters(
            original_action,
            flipped_action,
            action_target,
            original_weights=args.candidate_original_weights,
            regularization_cs=args.candidate_regularization_cs,
            outer_folds=args.selection_outer_folds,
            inner_folds=args.selection_inner_folds,
            seed=args.seed,
        )
        hyperparameter_selection["selection_split"] = (
            "+".join(fit_splits) + "_nested_oof"
        )
        args.original_weight = hyperparameter_selection["selected_original_weight"]
        args.regularization_c = hyperparameter_selection["selected_regularization_c"]
    mixed_action = args.original_weight * original_action + (1.0 - args.original_weight) * flipped_action
    fitted = fit_action_combo_oof(
        mixed_action,
        action_target,
        regularization_c=args.regularization_c,
        folds=args.folds,
        seed=args.seed,
    )
    stable_threshold_diagnostics = None
    if args.stable_action_thresholds:
        stable_threshold_diagnostics = fit_stable_label_thresholds(
            fitted["oof_action_probability"],
            action_target,
            folds=args.threshold_folds,
            seed=args.seed,
        )
        action_thresholds = stable_threshold_diagnostics["thresholds"]
    else:
        action_thresholds = fit_label_thresholds(fitted["oof_action_probability"], action_target)
    scaler, model = fitted["scaler"], fitted["model"]
    class_bits = ((model.classes_[:, None] & (1 << np.arange(action_target.shape[1]))) > 0).astype(float)

    original_test = payload["original"]["test"]["action_final"].numpy()
    flipped_test = payload["flip"]["test"]["action_final"].numpy()
    mixed_test = args.original_weight * original_test + (1.0 - args.original_weight) * flipped_test
    action_probability = model.predict_proba(scaler.transform(mixed_test)) @ class_bits
    test_action_target = payload["original"]["test"]["action_target"].numpy()

    reason_train_logits = concatenate(
        payload, "original", "reason_final", reason_fit_splits
    )
    reason_train_target = concatenate(
        payload, "original", "reason_target", reason_fit_splits
    )
    reason_train_probability = 1.0 / (1.0 + np.exp(-reason_train_logits))
    reason_threshold_diagnostics = None
    if args.reason_threshold_mode == "prior_anchored_train_oof":
        if args.reason_threshold_prior is None:
            raise ValueError("prior-anchored reason calibration requires a threshold prior")
        if args.reason_prior_alpha_step <= 0.0 or args.reason_prior_alpha_step > 1.0:
            raise ValueError("reason prior alpha step must be in (0,1]")
        alpha_grid = np.arange(
            0.0,
            1.0 + args.reason_prior_alpha_step * 0.5,
            args.reason_prior_alpha_step,
            dtype=np.float64,
        ).clip(0.0, 1.0)
        alpha_grid = np.unique(alpha_grid)
        reason_threshold_diagnostics = fit_prior_anchored_label_thresholds(
            reason_train_probability,
            reason_train_target,
            prior_thresholds=np.asarray(args.reason_threshold_prior, dtype=np.float64),
            alpha_grid=alpha_grid,
            folds=args.reason_threshold_folds,
            seed=args.seed,
            minimum_macro_gain=args.reason_prior_min_macro_gain,
        )
        reason_thresholds = reason_threshold_diagnostics["thresholds"]
    else:
        if args.reason_threshold_prior is not None:
            raise ValueError("reason threshold prior is only valid in prior-anchored mode")
        reason_thresholds = fit_label_thresholds(
            reason_train_probability, reason_train_target
        )
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
            "reason_threshold_mode": args.reason_threshold_mode,
            "reason_threshold_alpha": (
                None
                if reason_threshold_diagnostics is None
                else reason_threshold_diagnostics["selected_alpha"]
            ),
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
        "action_threshold_fit_split": f"{args.folds}-fold {'+'.join(fit_splits)} OOF",
        "reason_threshold_fit_split": "+".join(reason_fit_splits),
        "reason_threshold_mode": args.reason_threshold_mode,
        "reason_threshold_diagnostics": (
            None
            if reason_threshold_diagnostics is None
            else {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in reason_threshold_diagnostics.items()
            }
        ),
        "test_labels_used_for_parameters": False,
        "original_weight": args.original_weight,
        "regularization_c": args.regularization_c,
        "hyperparameter_selection": hyperparameter_selection,
        "stable_action_thresholds": args.stable_action_thresholds,
        "stable_threshold_diagnostics": (
            None
            if stable_threshold_diagnostics is None
            else {
                key: value.tolist()
                for key, value in stable_threshold_diagnostics.items()
            }
        ),
        "action_thresholds": action_thresholds.tolist(),
        "reason_thresholds": reason_thresholds.tolist(),
        "metrics": result,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "metrics_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
