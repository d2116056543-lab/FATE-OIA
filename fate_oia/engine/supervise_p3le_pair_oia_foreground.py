from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stream(cmd: list[str], cwd: Path, log_path: Path) -> int:
    print(json.dumps({"event": "foreground_command_start", "cmd": cmd}, ensure_ascii=False), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            text = line.rstrip()
            print(text, flush=True)
            log.write(text + "\n")
            log.flush()
        code = int(proc.wait())
        log.write(json.dumps({"event": "foreground_command_exit", "exit_code": code}, ensure_ascii=False) + "\n")
        return code


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground P3LE-PAIR-OIA V1 supervisor.")
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_p3le_pair_oia_v1.yaml")
    ap.add_argument("--output_root", default=".background_runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size1", type=int, default=3)
    ap.add_argument("--fallback_gradient_accumulation_steps1", type=int, default=11)
    ap.add_argument("--fallback_batch_size2", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps2", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--bdd_oia_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--require_review_pass", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"p3le_pair_oia_v1_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = output_dir / "supervisor_decisions.jsonl"
    append_jsonl(decisions, {"event": "supervisor_start", "output_dir": str(output_dir), "foreground": True, "background_forbidden": True})
    audit_cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.audit_p3le_pair_oia_implementation",
        "--config",
        args.config,
        "--output_dir",
        str(output_dir / "preflight_audit"),
        "--device",
        args.device,
        "--canonical_read_confirmed",
    ]
    code = run_stream(audit_cmd, root, output_dir / "audit.log")
    if code != 0:
        append_jsonl(decisions, {"event": "audit_failed", "exit_code": code})
        raise SystemExit(code)
    pass_file = output_dir / "preflight_audit" / "REVIEW_PASS_P3LE_PAIR_OIA_V1_1.txt"
    if args.require_review_pass and not pass_file.exists():
        append_jsonl(decisions, {"event": "review_pass_missing", "path": str(pass_file)})
        raise SystemExit(2)
    attempts = [
        (args.batch_size, args.gradient_accumulation_steps),
        (args.fallback_batch_size1, args.fallback_gradient_accumulation_steps1),
        (args.fallback_batch_size2, args.fallback_gradient_accumulation_steps2),
    ]
    last_code = 1
    for idx, (batch_size, grad_accum) in enumerate(attempts):
        run_dir = output_dir / f"train_b{batch_size}_a{grad_accum}"
        train_cmd = [
            sys.executable,
            "-m",
            "fate_oia.engine.train_p3le_pair_oia",
            "--config",
            args.config,
            "--output_dir",
            str(run_dir),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(batch_size),
            "--gradient_accumulation_steps",
            str(grad_accum),
            "--num_workers",
            str(args.num_workers),
            "--data_root",
            args.bdd_oia_root,
            "--raw_root",
            args.bdd_oia_root,
            "--bdd100k_root",
            args.bdd100k_root,
            "--max_train_samples",
            str(args.max_train_samples),
            "--max_test_samples",
            str(args.max_test_samples),
        ]
        append_jsonl(decisions, {"event": "training_attempt_start", "attempt": idx, "batch_size": batch_size, "grad_accum": grad_accum, "run_dir": str(run_dir)})
        last_code = run_stream(train_cmd, root, output_dir / f"train_attempt_{idx}.log")
        append_jsonl(decisions, {"event": "training_attempt_exit", "attempt": idx, "exit_code": last_code})
        if last_code == 0:
            goal = run_dir / "GOAL_COMPLETED_P3LE_PAIR_OIA_V1.json"
            if goal.exists():
                append_jsonl(decisions, {"event": "goal_completed", "run_dir": str(run_dir)})
                raise SystemExit(0)
        append_jsonl(decisions, {"event": "training_attempt_failed_try_fallback", "attempt": idx, "exit_code": last_code})
    (output_dir / "FAIL_P3LE_PAIR_OIA_V1.json").write_text(json.dumps({"last_exit_code": last_code}, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(last_code)


if __name__ == "__main__":
    main()
