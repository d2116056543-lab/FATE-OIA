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
