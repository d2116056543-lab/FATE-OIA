from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "experiment",
    "Act_mF1_fused",
    "Act_oF1_fused",
    "Exp_mF1",
    "Exp_oF1",
    "Exp_mAP",
    "LongTail_Exp_mF1",
    "grounding_hit",
    "deletion_drop",
    "sufficiency",
    "avg_tokens",
    "latency",
    "peak_mem",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig") or "{}")


def summarize_run(run_dir: Path) -> dict[str, Any]:
    epoch_dirs = sorted(p for p in run_dir.glob("epoch_*") if p.is_dir())
    epoch_dir = epoch_dirs[-1] if epoch_dirs else run_dir
    metrics = _load_json(epoch_dir / "metrics_summary.json")
    token_rows = []
    token_path = epoch_dir / "token_stats.jsonl"
    if token_path.exists():
        token_rows = [json.loads(line) for line in token_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    avg_tokens = None
    if token_rows:
        avg_tokens = sum(float(r.get("reduced_tokens", 0)) for r in token_rows) / max(len(token_rows), 1)
    return {
        "experiment": run_dir.name,
        "Act_mF1_fused": metrics.get("Act_mF1_fused"),
        "Act_oF1_fused": metrics.get("Act_oF1_fused"),
        "Exp_mF1": metrics.get("Exp_mF1"),
        "Exp_oF1": metrics.get("Exp_oF1"),
        "Exp_mAP": metrics.get("Exp_mAP"),
        "LongTail_Exp_mF1": metrics.get("long_tail_Exp_mF1"),
        "grounding_hit": metrics.get("pointing_game_hit_object"),
        "deletion_drop": metrics.get("deletion_drop"),
        "sufficiency": metrics.get("sufficiency"),
        "avg_tokens": avg_tokens,
        "latency": metrics.get("latency"),
        "peak_mem": metrics.get("peak_mem"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FATE-OIA ablation CSV from completed output directories.")
    ap.add_argument("--run_dirs", nargs="+", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()
    rows = [summarize_run(Path(p)) for p in args.run_dirs]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"event": "fate_oia_ablation_table", "rows": len(rows), "output_csv": str(out)}), flush=True)


if __name__ == "__main__":
    main()
