from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def append_decision(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stream(cmd: list[str], cwd: Path, decisions: Path) -> int:
    append_decision(decisions, {"event": "run_start", "cmd": cmd})
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    append_decision(decisions, {"event": "run_end", "returncode": rc})
    return rc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_egcaf_oia_v1.yaml")
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size1", type=int, default=3)
    ap.add_argument("--fallback_gradient_accumulation_steps1", type=int, default=11)
    ap.add_argument("--fallback_batch_size2", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps2", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default=r".background_runs\egcaf_oia_v1_full_28")
    ap.add_argument("--data_root", default=r"E:\sbw\BDD-OIA\data")
    ap.add_argument("--raw_root", default=r"E:\sbw\BDD-OIA")
    ap.add_argument("--bdd100k_root", default=r"E:\sbw\BDD100K")
    ap.add_argument("--print_every", type=int, default=200)
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--goal_mode", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    preflight = root / ".background_runs" / "egcaf_oia_v1_1_preflight" / "REVIEW_PASS_EGCAF_OIA_V1_1.txt"
    decisions = root / args.output_dir / "supervisor_decisions.jsonl"
    if args.require_review_pass and not preflight.exists():
        raise SystemExit(f"Missing required review pass: {preflight}")
    attempts = [
        (args.batch_size, args.gradient_accumulation_steps, "primary"),
        (args.fallback_batch_size1, args.fallback_gradient_accumulation_steps1, "fallback1"),
        (args.fallback_batch_size2, args.fallback_gradient_accumulation_steps2, "fallback2"),
    ]
    rc = 1
    for batch_size, accum, name in attempts:
        out_dir = args.output_dir if name == "primary" else f"{args.output_dir}_{name}_b{batch_size}a{accum}"
        cmd = [
            sys.executable, "-m", "fate_oia.engine.train_egcaf_oia",
            "--config", args.config,
            "--output_dir", out_dir,
            "--data_root", args.data_root,
            "--raw_root", args.raw_root,
            "--bdd100k_root", args.bdd100k_root,
            "--epochs", str(args.epochs),
            "--batch_size", str(batch_size),
            "--gradient_accumulation_steps", str(accum),
            "--device", args.device,
            "--no_feature_cache",
            "--test_only",
            "--print_every", str(args.print_every),
        ]
        append_decision(decisions, {"event": "attempt_start", "name": name, "batch_size": batch_size, "grad_accum": accum, "output_dir": out_dir})
        rc = stream(cmd, root, decisions)
        if rc == 0:
            args.output_dir = out_dir
            break
        append_decision(decisions, {"event": "attempt_failed", "name": name, "returncode": rc, "next_fallback": True})
    if rc != 0:
        raise SystemExit(rc)
    goal = root / args.output_dir / "GOAL_COMPLETED_EGCAF_OIA_V1.json"
    if args.goal_mode:
        goal.write_text(json.dumps({"completed": True, "epochs": args.epochs, "review_pass": str(preflight), "method": "EG-CAF-OIA V1.1"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
