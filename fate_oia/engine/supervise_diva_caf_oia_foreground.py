from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


class RequireReviewPass(RuntimeError):
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Foreground supervisor for DIVA-CAF-OIA V2")
    parser.add_argument("--review_pass", default=".background_runs/diva_caf_oia_v2_preflight/REVIEW_PASS_DIVA_CAF_OIA_V2.txt")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not Path(args.review_pass).exists():
        raise RequireReviewPass(f"missing review pass: {args.review_pass}")
    cmd = args.cmd if args.cmd else [sys.executable, "-m", "fate_oia.engine.train_diva_caf_oia"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        proc.terminate()
        raise
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
