from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from fate_oia.utils.cafe_artifacts import append_jsonl, write_json
from fate_oia.utils.cafe_review_gates import require_review_pass


def _run_stream(cmd: list[str], cwd: Path, status_path: Path) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last_epoch = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        event = {"timestamp": time.time(), "line": line.rstrip("\n")}
        try:
            payload = json.loads(line)
            if isinstance(payload, dict) and "epoch" in payload:
                last_epoch = payload.get("epoch")
                event["epoch"] = last_epoch
        except Exception:
            pass
        write_json(status_path, {"running": True, "last_epoch": last_epoch, "last_line": line.rstrip("\n"), "timestamp": time.time()})
    rc = proc.wait()
    write_json(status_path, {"running": False, "returncode": rc, "last_epoch": last_epoch, "timestamp": time.time()})
    return int(rc)


def _cmd(label: str, cmd: list[str], cwd: Path, decisions: Path) -> None:
    append_jsonl(decisions, {"event": "run_command", "label": label, "cmd": cmd})
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    append_jsonl(decisions, {"event": "command_result", "label": label, "returncode": proc.returncode, "tail": proc.stdout[-4000:]})
    print(proc.stdout, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_cafe_oia_v2.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--foreground", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--require_review_pass", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--skip_preflight", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    cwd = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    decisions = out / "supervisor_decisions.jsonl"
    status = out / "supervisor_live_status.json"
    if not args.foreground:
        raise RuntimeError("CAFE-OIA V2 supervisor only supports foreground execution.")
    for context_path in [Path("E:/sbw/FATE_Drive/task_plan.md"), Path("E:/sbw/FATE_Drive/findings.md"), Path("E:/sbw/FATE_Drive/progress.md")]:
        if not context_path.exists():
            raise FileNotFoundError(f"Required canonical context file missing: {context_path}")
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != "clean_cafe_oia_v1":
        raise RuntimeError(f"Expected branch clean_cafe_oia_v1, got {branch}")
    if "fate_oia_clean_cafe_oia_v1_worktree" not in str(cwd):
        raise RuntimeError(f"Expected clean_cafe worktree path, got {cwd}")
    py = sys.executable
    if not args.skip_preflight:
        _cmd("py_compile", [py, "-m", "py_compile",
            "fate_oia/engine/train_cafe_oia.py",
            "fate_oia/engine/supervise_cafe_oia_v2_foreground.py",
            "fate_oia/engine/audit_cafe_oia_v2_implementation.py",
            "fate_oia/engine/audit_cafe_evidence_cache.py",
            "fate_oia/engine/calibrate_cafe_oia.py",
            "fate_oia/models/cafe_oia_model.py",
            "fate_oia/models/causal_evidence_pooler.py",
            "fate_oia/models/counterfactual_evidence_intervention.py",
            "fate_oia/models/evidence_memory_bank.py",
            "fate_oia/losses/counterfactual_direct_effect_v2.py",
            "fate_oia/utils/config_io.py",
            "fate_oia/utils/plateau_rollback.py"], cwd, decisions)
        _cmd("targeted_pytest", [py, "-m", "pytest",
            "tests/test_cafe_v2_config_loading.py",
            "tests/test_cafe_v2_evidence_pooler.py",
            "tests/test_cafe_v2_counterfactual_intervention.py",
            "tests/test_cafe_v2_calibration.py",
            "tests/test_cafe_v2_plateau_restore.py",
            "tests/test_cafe_v2_review_gates.py",
            "tests/test_cafe_v2_artifacts.py", "-q"], cwd, decisions)
        _cmd("implementation_audit", [py, "-m", "fate_oia.engine.audit_cafe_oia_v2_implementation", "--config", args.config, "--output_dir", ".background_runs/cafe_oia_v2_preflight", "--device", args.device], cwd, decisions)
    if args.require_review_pass:
        require_review_pass(".background_runs/cafe_oia_v2_preflight/REVIEW_PASS_CAFE_V2.txt")
    train_cmd = [
        py,
        "-u",
        "-m",
        "fate_oia.engine.train_cafe_oia",
        "--config",
        args.config,
        "--output_dir",
        str(out),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--device",
        args.device,
    ]
    append_jsonl(decisions, {"event": "formal_training_start", "cmd": train_cmd})
    rc = _run_stream(train_cmd, cwd, status)
    append_jsonl(decisions, {"event": "formal_training_exit", "returncode": rc})
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
