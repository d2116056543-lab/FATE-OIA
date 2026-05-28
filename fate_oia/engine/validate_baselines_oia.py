from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.eval_score_calibrated import evaluate_score_logits
from fate_oia.utils.config_fingerprint import write_fingerprint


RUN_C_REFERENCE = {"joint": 0.547844, "Act_mF1": 0.714387, "Exp_mF1": 0.381301, "Exp_mAP": 0.367822}
P1_REFERENCE = {"joint": 0.551597, "Act_mF1": 0.714387, "Exp_mF1": 0.388808, "Exp_mAP": 0.367822}


def compute_cached_metrics(action_logits: torch.Tensor, reason_logits: torch.Tensor, labels_action: torch.Tensor, labels_reason: torch.Tensor) -> dict[str, float]:
    result = evaluate_score_logits(action_logits, reason_logits, labels_action, labels_reason, threshold_mode="fixed")
    metrics = result["metrics"]
    return {
        "joint": float(result["joint"]),
        "Act_mF1": float(metrics["Act_mF1"]),
        "Act_oF1": float(metrics["Act_oF1"]),
        "Exp_mF1": float(metrics["Exp_mF1"]),
        "Exp_oF1": float(metrics["Exp_oF1"]),
        "Exp_mAP": float(metrics["Exp_mAP"]),
    }


def _load_run_c_epoch(run_c_dir: Path, epoch: int = 14) -> dict[str, torch.Tensor]:
    epoch_dir = run_c_dir / f"epoch_{epoch:03d}"
    return {
        "action_logits": torch.load(epoch_dir / "logits_action_fused_test.pt", map_location="cpu"),
        "reason_logits": torch.load(epoch_dir / "logits_reason_test.pt", map_location="cpu"),
        "labels_action": torch.load(epoch_dir / "labels_action_test.pt", map_location="cpu"),
        "labels_reason": torch.load(epoch_dir / "labels_reason_test.pt", map_location="cpu"),
    }


def _close_enough(actual: dict[str, float], expected: dict[str, float], tolerance: float) -> dict[str, Any]:
    return {
        key: {
            "actual": float(actual[key]),
            "expected": float(expected[key]),
            "abs_diff": abs(float(actual[key]) - float(expected[key])),
            "ok": abs(float(actual[key]) - float(expected[key])) <= tolerance,
        }
        for key in expected
    }


def validate(run_c_dir: Path, tail_adapter_dir: Path | None, output_dir: Path, tolerance: float = 1e-4) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_run_c_epoch(run_c_dir)
    run_c = compute_cached_metrics(**cached)
    checks = _close_enough(run_c, RUN_C_REFERENCE, tolerance)
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "run_c_dir": str(run_c_dir),
        "run_c_metrics": run_c,
        "run_c_checks": checks,
        "run_c_ok": all(v["ok"] for v in checks.values()),
    }
    if tail_adapter_dir and (tail_adapter_dir / "metrics_summary.jsonl").exists():
        p1_rows = [json.loads(line) for line in (tail_adapter_dir / "metrics_summary.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        p1 = next((row for row in p1_rows if row.get("stage") == "P1" and int(row.get("epoch", -1)) == 1), None)
        if p1:
            p1_metrics = {k: float(p1[k]) for k in ("joint", "Act_mF1", "Exp_mF1", "Exp_mAP")}
            payload["p1_metrics"] = p1_metrics
            payload["p1_checks"] = _close_enough(p1_metrics, P1_REFERENCE, 1e-3)
    (output_dir / "baseline_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "run_c_reproduced_metrics.json").write_text(json.dumps(run_c, ensure_ascii=False, indent=2), encoding="utf-8")
    write_fingerprint(output_dir / "config_fingerprint.json", payload)
    if not payload["run_c_ok"]:
        raise RuntimeError(f"Run C cached metrics drifted beyond tolerance {tolerance}: {checks}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate cached Run C/P1 FATE-OIA baselines before new research runs.")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--tail_adapter_dir", default=".background_runs/fate_oia_tail_adapter_20260528_foreground")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()
    out = Path(args.output_dir) if args.output_dir else Path(".background_runs") / f"baseline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    payload = validate(Path(args.run_c_dir), Path(args.tail_adapter_dir) if args.tail_adapter_dir else None, out, args.tolerance)
    print(json.dumps({"event": "baseline_validation", "output_dir": str(out), "run_c_ok": payload["run_c_ok"], "run_c_metrics": payload["run_c_metrics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
