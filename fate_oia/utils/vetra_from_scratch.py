from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def validate_clean_stage1_command(command: Sequence[str]) -> None:
    lowered = [str(value).lower() for value in command]
    for forbidden in ("--init-model-checkpoint", "--resume"):
        if forbidden in lowered:
            raise ValueError(f"clean stage1 forbids {forbidden}")


def validate_internal_stage_checkpoint(run_root: str | Path, checkpoint: str | Path) -> Path:
    root = Path(run_root).resolve()
    stage1 = (root / "stage1_aie").resolve()
    candidate = Path(checkpoint).resolve()
    try:
        candidate.relative_to(stage1)
    except ValueError as error:
        raise ValueError("stage2 checkpoint is outside current clean run") from error
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def action_code(target: np.ndarray) -> np.ndarray:
    return (target.astype(np.int64) * (1 << np.arange(target.shape[1]))).sum(axis=1)


def _combo_membership(classes: np.ndarray, action_dim: int) -> np.ndarray:
    return ((classes[:, None] & (1 << np.arange(action_dim))) > 0).astype(np.float64)


def _fit_combo(train_logits: np.ndarray, target: np.ndarray, regularization_c: float):
    scaler = StandardScaler().fit(train_logits)
    model = LogisticRegression(C=regularization_c, max_iter=3000)
    model.fit(scaler.transform(train_logits), action_code(target))
    return scaler, model


def fit_action_combo_oof(
    train_logits: np.ndarray,
    target: np.ndarray,
    *,
    regularization_c: float = 10.0,
    folds: int = 5,
    seed: int = 20260815,
) -> dict[str, object]:
    train_logits = np.asarray(train_logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if train_logits.ndim != 2 or target.shape != train_logits.shape:
        raise ValueError("train logits and targets must be matching [N,A] arrays")
    if len(train_logits) < folds:
        raise ValueError("not enough rows for requested OOF folds")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(target, dtype=np.float64)
    for fit_indices, held_indices in splitter.split(train_logits):
        scaler, model = _fit_combo(train_logits[fit_indices], target[fit_indices], regularization_c)
        combo_probability = model.predict_proba(scaler.transform(train_logits[held_indices]))
        oof[held_indices] = combo_probability @ _combo_membership(model.classes_, target.shape[1])

    scaler, model = _fit_combo(train_logits, target, regularization_c)
    return {
        "scaler": scaler,
        "model": model,
        "oof_action_probability": oof,
    }


def fit_label_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    minimum: float = 0.01,
    maximum: float = 0.99,
    steps: int = 197,
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if probability.ndim != 2 or target.shape != probability.shape:
        raise ValueError("probabilities and targets must be matching [N,L] arrays")
    grid = np.linspace(minimum, maximum, steps, dtype=np.float64)
    thresholds = []
    for label in range(target.shape[1]):
        scores = [
            f1_score(target[:, label], probability[:, label] >= threshold, zero_division=0)
            for threshold in grid
        ]
        thresholds.append(grid[int(np.argmax(scores))])
    return np.asarray(thresholds, dtype=np.float32)


def fit_stable_label_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    folds: int = 10,
    seed: int = 20260815,
) -> dict[str, np.ndarray]:
    """Reduce threshold variance with train-only jackknife medians."""
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if probability.ndim != 2 or target.shape != probability.shape:
        raise ValueError("probabilities and targets must be matching [N,L] arrays")
    if folds < 2 or len(probability) < folds:
        raise ValueError("invalid threshold fold configuration")
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_thresholds = np.stack(
        [
            fit_label_thresholds(probability[fit_indices], target[fit_indices])
            for fit_indices, _ in splitter.split(probability)
        ]
    )
    return {
        "thresholds": np.median(fold_thresholds, axis=0).astype(np.float32),
        "fold_thresholds": fold_thresholds.astype(np.float32),
        "full_sample_thresholds": fit_label_thresholds(probability, target),
        "threshold_iqr": (
            np.percentile(fold_thresholds, 75, axis=0)
            - np.percentile(fold_thresholds, 25, axis=0)
        ).astype(np.float32),
    }


def fit_prior_anchored_label_thresholds(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    prior_thresholds: np.ndarray,
    alpha_grid: np.ndarray,
    folds: int = 5,
    seed: int = 20260815,
    minimum_macro_gain: float = 0.001,
) -> dict[str, object]:
    """Conservatively update a fixed threshold prior using train-only OOF evidence.

    The smallest blend that clears the requested OOF macro-F1 gain is used. This
    preserves the prior deployment prevalence unless current training rows prove
    that a larger update is useful, and never reads evaluation labels.
    """
    probability = np.asarray(probability, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prior = np.asarray(prior_thresholds, dtype=np.float64)
    alphas = np.asarray(alpha_grid, dtype=np.float64)
    if probability.ndim != 2 or target.shape != probability.shape:
        raise ValueError("probabilities and targets must be matching [N,L] arrays")
    if prior.shape != (target.shape[1],):
        raise ValueError("prior thresholds must contain one value per label")
    if folds < 2 or len(probability) < folds:
        raise ValueError("invalid threshold fold configuration")
    if alphas.ndim != 1 or len(alphas) == 0 or np.any(np.diff(alphas) < 0):
        raise ValueError("alpha grid must be a non-empty sorted vector")
    if not np.isclose(alphas[0], 0.0) or np.any((alphas < 0.0) | (alphas > 1.0)):
        raise ValueError("alpha grid must start at zero and remain in [0,1]")
    if minimum_macro_gain < 0.0:
        raise ValueError("minimum macro gain must be non-negative")

    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_data = [
        (
            held_indices,
            fit_label_thresholds(probability[fit_indices], target[fit_indices]),
        )
        for fit_indices, held_indices in splitter.split(probability)
    ]
    candidate_scores: list[dict[str, float]] = []
    for alpha in alphas:
        oof_prediction = np.zeros_like(target, dtype=bool)
        for held_indices, fold_thresholds in fold_data:
            thresholds = alpha * fold_thresholds + (1.0 - alpha) * prior
            oof_prediction[held_indices] = probability[held_indices] >= thresholds
        macro_f1 = float(
            np.mean(
                [
                    f1_score(
                        target[:, label],
                        oof_prediction[:, label],
                        zero_division=0,
                    )
                    for label in range(target.shape[1])
                ]
            )
        )
        overall_f1 = float(
            f1_score(target.ravel(), oof_prediction.ravel(), zero_division=0)
        )
        candidate_scores.append(
            {
                "alpha": float(alpha),
                "macro_f1": macro_f1,
                "overall_f1": overall_f1,
            }
        )

    baseline = candidate_scores[0]
    eligible = [
        row
        for row in candidate_scores
        if row["macro_f1"] >= baseline["macro_f1"] + minimum_macro_gain
        and row["overall_f1"] >= baseline["overall_f1"]
    ]
    selected = min(eligible, key=lambda row: row["alpha"]) if eligible else baseline
    full_sample_thresholds = fit_label_thresholds(probability, target)
    selected_alpha = float(selected["alpha"])
    thresholds = (
        selected_alpha * full_sample_thresholds
        + (1.0 - selected_alpha) * prior
    ).astype(np.float32)
    return {
        "thresholds": thresholds,
        "selected_alpha": selected_alpha,
        "prior_thresholds": prior.astype(np.float32),
        "full_sample_thresholds": full_sample_thresholds,
        "minimum_macro_gain": float(minimum_macro_gain),
        "selection_rule": "smallest_alpha_with_oof_macro_gain_and_no_overall_drop",
        "candidate_scores": candidate_scores,
    }


def select_action_combo_hyperparameters(
    original_logits: np.ndarray,
    flipped_logits: np.ndarray,
    target: np.ndarray,
    *,
    original_weights: Sequence[float],
    regularization_cs: Sequence[float],
    outer_folds: int = 5,
    inner_folds: int = 4,
    seed: int = 20260815,
) -> dict[str, object]:
    """Select deployment hyperparameters using nested train-only OOF scores."""
    original_logits = np.asarray(original_logits, dtype=np.float64)
    flipped_logits = np.asarray(flipped_logits, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if original_logits.ndim != 2 or original_logits.shape != target.shape:
        raise ValueError("original logits and targets must be matching [N,A] arrays")
    if flipped_logits.shape != original_logits.shape:
        raise ValueError("original and flipped logits must have matching shapes")
    weights = tuple(float(value) for value in original_weights)
    regularizations = tuple(float(value) for value in regularization_cs)
    if not weights or not regularizations:
        raise ValueError("candidate grid must not be empty")
    if len(original_logits) < outer_folds or outer_folds < 2 or inner_folds < 2:
        raise ValueError("invalid nested fold configuration")

    outer = KFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    candidates: list[dict[str, object]] = []
    for original_weight in weights:
        mixed = original_weight * original_logits + (1.0 - original_weight) * flipped_logits
        for regularization_c in regularizations:
            fold_scores = []
            for fold, (fit_indices, held_indices) in enumerate(outer.split(mixed)):
                fold_inner_folds = min(inner_folds, len(fit_indices))
                fitted = fit_action_combo_oof(
                    mixed[fit_indices],
                    target[fit_indices],
                    regularization_c=regularization_c,
                    folds=fold_inner_folds,
                    seed=seed + fold + 1,
                )
                thresholds = fit_label_thresholds(
                    fitted["oof_action_probability"], target[fit_indices]
                )
                scaler, model = fitted["scaler"], fitted["model"]
                held_combo = model.predict_proba(scaler.transform(mixed[held_indices]))
                held_probability = held_combo @ _combo_membership(
                    model.classes_, target.shape[1]
                )
                held_prediction = held_probability >= thresholds[None, :]
                macro_f1 = float(
                    np.mean(
                        [
                            f1_score(
                                target[held_indices, label],
                                held_prediction[:, label],
                                zero_division=0,
                            )
                            for label in range(target.shape[1])
                        ]
                    )
                )
                overall_f1 = float(
                    f1_score(
                        target[held_indices].ravel(),
                        held_prediction.ravel(),
                        zero_division=0,
                    )
                )
                fold_scores.append(
                    {
                        "fold": fold,
                        "macro_f1": macro_f1,
                        "overall_f1": overall_f1,
                        "selection_score": 0.8 * macro_f1 + 0.2 * overall_f1,
                    }
                )
            candidates.append(
                {
                    "original_weight": original_weight,
                    "regularization_c": regularization_c,
                    "mean_macro_f1": float(np.mean([row["macro_f1"] for row in fold_scores])),
                    "mean_overall_f1": float(np.mean([row["overall_f1"] for row in fold_scores])),
                    "mean_selection_score": float(
                        np.mean([row["selection_score"] for row in fold_scores])
                    ),
                    "fold_scores": fold_scores,
                }
            )

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["mean_selection_score"],
            -row["mean_macro_f1"],
            -row["mean_overall_f1"],
            abs(row["original_weight"] - 0.75),
            row["regularization_c"],
        ),
    )
    best = ranked[0]
    return {
        "selection_split": "provided_train_rows_nested_oof",
        "selection_objective": "0.8*macro_f1+0.2*overall_f1",
        "selected_original_weight": best["original_weight"],
        "selected_regularization_c": best["regularization_c"],
        "candidate_scores": ranked,
    }
