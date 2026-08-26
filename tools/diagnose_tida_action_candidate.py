import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ACTION_NAMES = ("forward", "stop", "left", "right")


def load_tensor(epoch_dir: Path, name: str) -> np.ndarray:
    value = torch.load(epoch_dir / f"{name}_test.pt", map_location="cpu")
    return value.detach().float().numpy()


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(target, score)) if np.unique(target).size == 2 else float("nan")


def metrics(target: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, float]:
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    prediction = probability >= threshold
    return {
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "ap": float(average_precision_score(target, probability)),
        "auc": safe_auc(target, probability),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("epoch_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    epoch_dir = args.epoch_dir

    base = load_tensor(epoch_dir, "pre_object_intent_action")
    candidate = load_tensor(epoch_dir, "object_intent_action_candidate")
    target = load_tensor(epoch_dir, "action_target").astype(np.int64)
    directional = load_tensor(epoch_dir, "object_intent_action_directional_utility_gate")
    risk = load_tensor(epoch_dir, "object_intent_action_risk_utility_gate")
    deployed = load_tensor(epoch_dir, "object_intent_action_delta")

    with (epoch_dir / "calibration.json").open("r", encoding="utf-8") as handle:
        calibration = json.load(handle)
    threshold = np.asarray(
        calibration.get("action_thresholds")
        or calibration.get("thresholds_action")
        or calibration.get("threshold_action")
        or [0.59, 0.66, 0.60, 0.61],
        dtype=np.float64,
    )
    if threshold.size != 4:
        threshold = np.asarray([0.59, 0.66, 0.60, 0.61], dtype=np.float64)

    scales = np.asarray([0, 1, 2, 4, 8, 16, 24, 32, 48, 64], dtype=np.float64)
    cutoffs = np.asarray([0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
    rows = []
    base_f1 = []
    deployed_f1 = []
    oracle_f1 = []
    for action_id, name in enumerate(ACTION_NAMES):
        y = target[:, action_id]
        z = base[:, action_id]
        d = candidate[:, action_id]
        base_metrics = metrics(y, z, float(threshold[action_id]))
        deployed_metrics = metrics(y, z + deployed[:, action_id], float(threshold[action_id]))
        base_f1.append(base_metrics["f1"])
        deployed_f1.append(deployed_metrics["f1"])

        best = {"f1": base_metrics["f1"], "scale": 0.0, "cutoff": 0.0, "source": "off"}
        source_stats = {}
        for source_name, utility in (("directional", directional), ("risk", risk)):
            source_best = dict(best)
            for scale in scales:
                for cutoff in cutoffs:
                    selected = utility[:, action_id] >= cutoff
                    delta = np.clip(d * scale, -0.18, 0.18) * selected
                    score = metrics(y, z + delta, float(threshold[action_id]))
                    if score["f1"] > source_best["f1"] + 1e-12:
                        source_best = {
                            "f1": score["f1"], "scale": float(scale),
                            "cutoff": float(cutoff), "source": source_name,
                        }
                    if score["f1"] > best["f1"] + 1e-12:
                        best = dict(source_best)
            source_stats[source_name] = source_best
        oracle_f1.append(best["f1"])

        probability = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
        near = np.abs(probability - threshold[action_id]) <= 0.10
        desired_sign = np.where(y > 0, 1.0, -1.0)
        sign_correct = desired_sign * d > 0
        rows.append({
            "action_id": action_id,
            "action": name,
            "positive_count": int(y.sum()),
            "threshold": float(threshold[action_id]),
            "base": base_metrics,
            "deployed": deployed_metrics,
            "test_oracle": best,
            "oracle_gain": float(best["f1"] - base_metrics["f1"]),
            "source_oracles": source_stats,
            "candidate_sign_accuracy": float(sign_correct.mean()),
            "near_boundary_count": int(near.sum()),
            "near_boundary_sign_accuracy": float(sign_correct[near].mean()) if near.any() else None,
            "candidate_rms": float(np.sqrt(np.mean(d * d))),
            "deployed_delta_rms": float(np.sqrt(np.mean(deployed[:, action_id] ** 2))),
        })

    summary = {
        "epoch_dir": str(epoch_dir),
        "thresholds": threshold.tolist(),
        "base_macro_f1": float(np.mean(base_f1)),
        "deployed_macro_f1": float(np.mean(deployed_f1)),
        "independent_test_oracle_macro_f1": float(np.mean(oracle_f1)),
        "independent_test_oracle_gain": float(np.mean(oracle_f1) - np.mean(base_f1)),
        "actions": rows,
        "interpretation": {
            "candidate_capacity_limited": bool(np.mean(oracle_f1) < 0.79),
            "policy_limited": bool(np.mean(oracle_f1) >= 0.79 and np.mean(deployed_f1) < 0.79),
        },
    }
    output = args.output or epoch_dir / "action_candidate_oracle_diagnostic.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
