from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8-sig"))


def summarize(alpha_json: str, threshold_json: str, failure_json: str, output_json: str) -> dict:
    alpha = load_json(alpha_json)
    threshold = load_json(threshold_json)
    failure = load_json(failure_json)
    rows = alpha.get("rows", [])
    best_alpha_row = max(rows, key=lambda r: r.get("joint", -999)) if rows else {}
    alpha0 = next((r for r in rows if abs(float(r.get("alpha", -1))) < 1e-12), {})
    delta = float(best_alpha_row.get("joint", 0.0)) - float(alpha0.get("joint", 0.0)) if best_alpha_row and alpha0 else 0.0
    threshold_gain = (
        threshold.get("threshold_gain_exp_mF1")
        or threshold.get("gain_exp_mF1")
        or threshold.get("per_label_gain_mF1")
        or 0.0
    )
    if not threshold_gain and "fixed" in threshold and "per_label" in threshold:
        fixed_metrics = threshold.get("fixed", {}).get("metrics", {})
        per_metrics = threshold.get("per_label", {}).get("metrics", {})
        fixed_exp = fixed_metrics.get("Exp_mF1") or fixed_metrics.get("Exp_mF1".replace("Exp", "Exp_")) or 0.0
        per_exp = per_metrics.get("Exp_mF1") or per_metrics.get("Exp_mF1".replace("Exp", "Exp_")) or 0.0
        # Existing threshold_sweep prefixes keys as Exp_mF1, Exp_mAP, etc.;
        # keep the fallback explicit for old artifacts.
        threshold_gain = float(per_exp or 0.0) - float(fixed_exp or 0.0)
    tail_mean_ap = float(failure.get("tail_mean_AP", 0.0) or 0.0)
    tail_best_f1 = float(failure.get("tail_best_possible_F1", failure.get("tail_mean_best_F1", 0.0)) or 0.0)
    out = {
        "best_alpha": best_alpha_row.get("alpha"),
        "best_joint": best_alpha_row.get("joint"),
        "delta_vs_alpha0": delta,
        "fusion_fix_priority": bool((best_alpha_row.get("alpha", 0.0) or 0.0) >= 0.10 and delta >= 0.002),
        "threshold_gain_exp_mF1": float(threshold_gain),
        "calibration_is_major_bottleneck": float(threshold_gain) >= 0.015,
        "tail_mean_AP": tail_mean_ap,
        "tail_best_possible_F1": tail_best_f1,
        "tail_representation_problem": tail_mean_ap < 0.15 and tail_best_f1 < 0.15,
        "tail_calibration_problem": tail_mean_ap >= 0.15 and float(threshold_gain) >= 0.015,
        "top_failed_reason_indices": failure.get("top_failed_reason_indices", []),
    }
    Path(output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize EviS score-branch offline diagnostics.")
    ap.add_argument("--alpha_json", required=True)
    ap.add_argument("--threshold_json", required=True)
    ap.add_argument("--failure_json", required=True)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()
    print(json.dumps(summarize(args.alpha_json, args.threshold_json, args.failure_json, args.output_json), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
