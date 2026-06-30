from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    args = ap.parse_args()
    cmd = [sys.executable, "-u", "-m", "fate_oia.engine.train_acpr_ntmcal_oia", "--config", "configs/fate_oia_train_360x640_acpr_ntmcal_v1.yaml", "--output_dir", ".background_runs/acpr_ntmcal_v1_360x640_testprimary", "--epochs", str(args.epochs), "--test_only", "--no_feature_cache", "--token_compression", "none", "--require_review_pass", ".background_runs/acpr_ntmcal_v1_preflight/REVIEW_PASS_ACPR_NTMCAL_V1.txt"]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
