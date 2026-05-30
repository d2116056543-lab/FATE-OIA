from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.utils.cafe_artifacts import write_json


def _macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    tp = (pred * labels).sum(0)
    fp = (pred * (1.0 - labels)).sum(0)
    fn = ((1.0 - pred) * labels).sum(0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)
    return float(f1.mean().item())


def fit_classwise_bias_temperature(
    reason_logits: torch.Tensor,
    reason_labels: torch.Tensor,
    optimize: str = "macro_f1",
    bias_values: torch.Tensor | None = None,
    temp_values: torch.Tensor | None = None,
) -> dict[str, Any]:
    reason_logits = reason_logits.detach().float().cpu()
    reason_labels = reason_labels.detach().float().cpu()
    reason_dim = reason_logits.shape[1]
    bias_grid = bias_values if bias_values is not None else torch.linspace(-1.5, 1.5, 31)
    temp_grid = temp_values if temp_values is not None else torch.linspace(0.70, 1.60, 10)
    best_temp = torch.ones(reason_dim)
    best_bias = torch.zeros(reason_dim)
    for r in range(reason_dim):
        best_score = -1.0
        x = reason_logits[:, r : r + 1]
        y = reason_labels[:, r : r + 1]
        for temp in temp_grid:
            for bias in bias_grid:
                score = _macro_f1_from_logits(x / float(temp) + float(bias), y)
                if score > best_score:
                    best_score = score
                    best_temp[r] = float(temp)
                    best_bias[r] = float(bias)
    calibrated = apply_calibration(reason_logits, {"temperature": best_temp.tolist(), "bias": best_bias.tolist()})
    return {
        "type": "classwise_bias_temperature",
        "optimize": optimize,
        "temperature": best_temp.tolist(),
        "bias": best_bias.tolist(),
        "raw_macro_f1": _macro_f1_from_logits(reason_logits, reason_labels),
        "calibrated_macro_f1": _macro_f1_from_logits(calibrated, reason_labels),
    }


def apply_calibration(reason_logits: torch.Tensor, params: dict[str, Any]) -> torch.Tensor:
    temp = torch.tensor(params.get("temperature", [1.0] * reason_logits.shape[1]), device=reason_logits.device, dtype=reason_logits.dtype)
    bias = torch.tensor(params.get("bias", [0.0] * reason_logits.shape[1]), device=reason_logits.device, dtype=reason_logits.dtype)
    return reason_logits / temp.clamp_min(1e-4).view(1, -1) + bias.view(1, -1)


def threshold_sweep_global(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    best = {"threshold": 0.5, "macro_f1": -1.0}
    for th in torch.linspace(0.05, 0.95, 19):
        score = _macro_f1_from_logits(logits, labels, float(th))
        if score > best["macro_f1"]:
            best = {"threshold": float(th), "macro_f1": score}
    return best


def threshold_sweep_per_label(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    thresholds: list[float] = []
    f1s: list[float] = []
    for r in range(logits.shape[1]):
        best_th = 0.5
        best_f1 = -1.0
        for th in torch.linspace(0.05, 0.95, 19):
            score = _macro_f1_from_logits(logits[:, r : r + 1], labels[:, r : r + 1], float(th))
            if score > best_f1:
                best_f1 = score
                best_th = float(th)
        thresholds.append(best_th)
        f1s.append(best_f1)
    return {"thresholds": thresholds, "per_label_f1": f1s, "macro_f1": float(sum(f1s) / max(1, len(f1s)))}


def combined_metrics(action_logits: torch.Tensor, reason_logits: torch.Tensor, action_labels: torch.Tensor, reason_labels: torch.Tensor, action_dim: int = 4) -> dict[str, Any]:
    logits = torch.cat([action_logits, reason_logits], dim=1)
    labels = torch.cat([action_labels, reason_labels], dim=1)
    return evaluate_snna25(logits, labels, action_dim, threshold_mode="fixed", fixed_threshold=0.5)["metrics"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason_logits", required=True)
    ap.add_argument("--reason_labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    logits = torch.load(args.reason_logits, map_location="cpu")
    labels = torch.load(args.reason_labels, map_location="cpu")
    params = fit_classwise_bias_temperature(logits, labels)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, params)
    print(json.dumps(params))


if __name__ == "__main__":
    main()
