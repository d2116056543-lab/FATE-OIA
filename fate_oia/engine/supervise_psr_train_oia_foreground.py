from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_stream(cmd: list[str], cwd: Path) -> int:
    print(json.dumps({"event": "foreground_command_start", "cmd": cmd}, ensure_ascii=False), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip(), flush=True)
    return int(proc.wait())


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground PSR-Train OIA V1 supervisor.")
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_psr_train_oia_v1.yaml")
    ap.add_argument("--output_root", default=".background_runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--require_audit", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"psr_train_oia_v1_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = output_dir / "supervisor_decisions.jsonl"
    decisions.write_text(json.dumps({"event": "supervisor_start", "output_dir": str(output_dir), "foreground": True, "background_forbidden": True}, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.require_audit:
        audit_cmd = [
            sys.executable,
            "-m",
            "fate_oia.engine.audit_psr_train_oia_implementation",
            "--config",
            args.config,
            "--output_dir",
            str(output_dir / "preflight_audit"),
        ]
        code = run_stream(audit_cmd, root)
        if code != 0:
            decisions.write_text(decisions.read_text(encoding="utf-8") + json.dumps({"event": "audit_failed", "exit_code": code}, ensure_ascii=False) + "\n", encoding="utf-8")
            raise SystemExit(code)
    train_cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_psr_train_oia",
        "--config",
        args.config,
        "--output_dir",
        str(output_dir),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_workers",
        str(args.num_workers),
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_test_samples",
        str(args.max_test_samples),
    ]
    code = run_stream(train_cmd, root)
    decisions.write_text(decisions.read_text(encoding="utf-8") + json.dumps({"event": "training_exit", "exit_code": code}, ensure_ascii=False) + "\n", encoding="utf-8")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
