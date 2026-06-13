from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    if args.require_review_pass and not Path(".background_runs/eagle_pu_v1_preflight/REVIEW_PASS_EAGLE_PU_V1.txt").exists():
        raise SystemExit("Missing REVIEW_PASS_EAGLE_PU_V1.txt")
    cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_eagle_pu_oia", "--config", args.config, "--output_dir", args.output_dir, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--device", args.device, "--test_only", "--no_feature_cache", "--require_no_token_compression"]
    print("foreground_cmd=" + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    raise SystemExit(proc.wait())

if __name__ == "__main__":
    main()
