from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_stream(cmd: list[str], label: str) -> None:
    print(json.dumps({"event": "gem_supervisor_step_start", "label": label, "cmd": cmd}), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    print(json.dumps({"event": "gem_supervisor_step_end", "label": label, "returncode": code}), flush=True)
    if code != 0:
        raise SystemExit(code)


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def require_clean_pushed_head() -> None:
    status = git_text(["status", "--short"])
    if status:
        raise SystemExit(f"worktree is not clean before formal training:\n{status}")
    local = git_text(["rev-parse", "HEAD"])
    remote = git_text(["ls-remote", "github", "refs/heads/acpr_gem_v1"]).split()[0]
    if local != remote:
        raise SystemExit(f"local HEAD {local} != GitHub acpr_gem_v1 {remote}")


def require_review_pass() -> None:
    paths = [
        Path(".background_runs/acpr_gem_v1_preflight_exact_head/REVIEW_PASS_ACPR_GEM_V1.txt"),
        Path(".background_runs/acpr_gem_v1_preflight/REVIEW_PASS_ACPR_GEM_V1.txt"),
    ]
    if not any(p.exists() for p in paths):
        raise SystemExit("missing REVIEW_PASS_ACPR_GEM_V1.txt")


def selected_batch_accum(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = data.get("selected")
    if not selected:
        raise SystemExit("memory probe did not select a stable batch/accum")
    return int(selected[0]), int(selected[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_gem_v1.yaml")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference_checkpoint", default="")
    parser.add_argument("--require_review_pass", action="store_true")
    args = parser.parse_args()
    if not args.reference_checkpoint:
        raise SystemExit("reference checkpoint is required for ACPR-GEM gates")
    require_clean_pushed_head()
    if args.require_review_pass:
        require_review_pass()

    py = sys.executable
    run_stream([
        py,
        "-m",
        "fate_oia.engine.audit_acpr_gem_implementation",
        "--config",
        args.config,
        "--output_dir",
        ".background_runs/acpr_gem_v1_preflight_exact_head",
        "--device",
        args.device,
        "--write_review_pass",
    ], "implementation_audit")
    run_stream([
        py,
        "-m",
        "fate_oia.engine.audit_acpr_gem_gates",
        "--config",
        args.config,
        "--reference_checkpoint",
        args.reference_checkpoint,
        "--output_dir",
        ".background_runs/acpr_gem_v1_gates",
        "--device",
        args.device,
    ], "gate_a_to_f")
    run_stream([
        py,
        "-m",
        "fate_oia.engine.probe_acpr_gem_memory",
        "--config",
        args.config,
        "--output_dir",
        ".background_runs/acpr_gem_v1_memory_probe",
        "--device",
        args.device,
        "--candidates",
        "6:5",
        "5:6",
        "4:8",
        "3:10",
        "2:15",
    ], "memory_probe")
    batch_size, grad_accum = selected_batch_accum(Path(".background_runs/acpr_gem_v1_memory_probe/GEM_MEMORY_PASS.json"))
    run_stream([
        py,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_oia",
        "--config",
        args.config,
        "--output_dir",
        ".background_runs/acpr_gem_v1_supervisor_smoke",
        "--epochs",
        "1",
        "--batch_size",
        "1",
        "--gradient_accumulation_steps",
        "2",
        "--max_train_samples",
        "4",
        "--max_test_samples",
        "4",
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ], "tiny_real_data_smoke")
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_oia",
        "--config",
        args.config,
        "--output_dir",
        ".background_runs/acpr_gem_v1_full",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ]
    run_stream(cmd, "formal_full_training")


if __name__ == "__main__":
    main()
