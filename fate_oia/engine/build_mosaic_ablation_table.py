from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fate_oia.utils.mosaic_artifacts import write_json


def build_row(run_dir: str | Path, *, epoch: int | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if epoch is None:
        epoch_dirs = sorted(run_dir.glob("epoch_*"))
        if not epoch_dirs:
            raise FileNotFoundError("no MOSAIC epoch artifacts found")
        epoch_dir = epoch_dirs[-1]
    else:
        epoch_dir = run_dir / f"epoch_{epoch:03d}"
    metrics = json.loads((epoch_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    action = json.loads((epoch_dir / "action_branch_metrics.json").read_text(encoding="utf-8"))
    reason = json.loads((epoch_dir / "reason_branch_metrics.json").read_text(encoding="utf-8"))
    visual_path = run_dir / "visual_audit" / "summary.json"
    visual = json.loads(visual_path.read_text(encoding="utf-8")) if visual_path.exists() else {}
    return {
        "run": run_dir.name,
        "epoch": metrics["epoch"],
        "Act_mF1_visual": action["visual"]["Act_mF1"],
        "Act_mF1_raw": metrics["raw"]["Act_mF1"],
        "Act_mF1_deploy": metrics["deploy_fixed"]["Act_mF1"],
        "Act_oF1_deploy": metrics["deploy_fixed"]["Act_oF1"],
        "Exp_mF1_latent": reason["latent"]["Exp_mF1"],
        "Exp_mF1_deploy": metrics["deploy_fixed"]["Exp_mF1"],
        "Exp_oF1_deploy": metrics["deploy_fixed"]["Exp_oF1"],
        "Exp_mAP": metrics["raw"]["Exp_mAP"],
        "deploy_joint": metrics["deploy_fixed"]["joint"],
        "factor_full": visual.get("full_factor_metric"),
        "factor_content_only": visual.get("content_only_factor_metric"),
        "factor_prior_only": visual.get("prior_only_factor_metric"),
        "posterior_recovery_pass": visual.get("posterior_recovery_pass"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    rows = [build_row(path) for path in args.run_dir]
    write_json(args.output_json, rows)
    with Path(args.output_csv).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
