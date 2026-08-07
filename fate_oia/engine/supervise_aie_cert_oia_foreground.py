from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=16); p.add_argument("--device", default="cuda"); p.add_argument("--require-review-pass", action="store_true")
    args, extra = p.parse_known_args(); review = Path(".review/aie_cert_oia_v1/REVIEW_PASS_AIE_CERT_OIA_V1.json")
    pilot = Path(".review/aie_cert_oia_v1/PILOT_PASS_AIE_CERT_OIA_V1.json")
    profile = Path(".review/aie_cert_oia_v1/AIE_CERT_RUNTIME_PROFILE.json")
    ready = Path(".review/aie_cert_oia_v1/AIE_CERT_FULL_TRAIN_READY.json")
    if args.require_review_pass and any(not path.exists() for path in (review, pilot, profile, ready)):
        raise SystemExit("AIE-CERT REVIEW_PASS, runtime profile, PILOT_PASS and FULL_TRAIN_READY are required")
    if args.require_review_pass:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        records = [json.loads(path.read_text(encoding="utf-8")) for path in (review, pilot, profile, ready)]
        if not all(record.get("pass") and record.get("git_head") == head for record in records):
            raise SystemExit("AIE-CERT gate binding does not match current clean HEAD")
        selected = records[2]["selected"]
        extra = ["--batch-size", str(selected["batch"]), "--gradient-accumulation-steps", str(selected["accum"]),
                 "--num-workers", str(selected["workers"]), *extra]
    command = [sys.executable, "-u", "-m", "fate_oia.engine.train_aie_cert_oia", "--config", args.config,
               "--output-dir", args.output_dir, "--run-kind", "full", "--epochs", str(args.epochs), "--device", args.device, *extra]
    print(json.dumps({"foreground_command": command}), flush=True)
    completed = subprocess.run(command)
    raise SystemExit(completed.returncode)


if __name__ == "__main__": main()
