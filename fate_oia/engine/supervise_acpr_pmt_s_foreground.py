from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("[pmt_supervisor]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    for p in [r"E:\sbw\FATE_Drive\task_plan.md", r"E:\sbw\FATE_Drive\findings.md", r"E:\sbw\FATE_Drive\progress.md"]:
        Path(p).read_text(encoding="utf-8", errors="ignore")
    if args.require_review_pass and not Path(".background_runs/acpr_pmt_s_v1_preflight/REVIEW_PASS_ACPR_PMT_S_V1.txt").exists():
        raise SystemExit("missing REVIEW_PASS_ACPR_PMT_S_V1.txt")
    run([py, "-u", "-m", "fate_oia.engine.train_acpr_oia", "--config", args.config, "--output_dir", args.output_dir, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"])


if __name__ == "__main__":
    main()
