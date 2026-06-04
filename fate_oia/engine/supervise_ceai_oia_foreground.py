from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def append_event(decisions: Path, row: dict) -> None:
    decisions.parent.mkdir(parents=True, exist_ok=True)
    with decisions.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_capture(cmd: list[str], decisions: Path, *, tag: str) -> tuple[int, str]:
    append_event(decisions, {"event": "run_start", "tag": tag, "cmd": cmd})
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = proc.wait()
    output = "".join(lines)
    append_event(decisions, {"event": "run_end", "tag": tag, "returncode": code})
    return code, output


def run(cmd: list[str], decisions: Path, *, tag: str) -> None:
    code, output = run_capture(cmd, decisions, tag=tag)
    if code != 0:
        append_event(decisions, {"event": "run_failed", "tag": tag, "returncode": code, "tail": output[-4000:]})
        raise RuntimeError(f"command failed code={code}: {' '.join(cmd)}")


def is_oom(output: str) -> bool:
    text = output.lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text or "cublas_status_alloc_failed" in text


def train_command(
    py: str,
    args: argparse.Namespace,
    output_dir: Path,
    *,
    batch_size: int,
    grad_accum: int,
    num_workers: int,
) -> list[str]:
    return [
        py,
        "-m",
        "fate_oia.engine.train_ceai_oia",
        "--config",
        args.config,
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(grad_accum),
        "--num_workers",
        str(num_workers),
        "--max_train_samples",
        "0",
        "--max_test_samples",
        "0",
        "--device",
        args.device,
        "--data_root",
        args.bdd_oia_root,
        "--raw_root",
        args.bdd_oia_root,
        "--bdd100k_root",
        args.bdd100k_root,
        "--best_selection_split",
        "test",
    ]


def run_full_with_oom_fallbacks(py: str, args: argparse.Namespace, decisions: Path, full: Path) -> Path:
    attempts = [
        {
            "name": "primary_b4_acc8",
            "batch_size": args.batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "output_dir": full,
        },
        {
            "name": "fallback_b3_acc11",
            "batch_size": args.fallback_batch_size1,
            "grad_accum": args.fallback_gradient_accumulation_steps1,
            "output_dir": full.with_name(full.name + "_b3"),
        },
        {
            "name": "fallback_b2_acc16",
            "batch_size": args.fallback_batch_size2,
            "grad_accum": args.fallback_gradient_accumulation_steps2,
            "output_dir": full.with_name(full.name + "_b2"),
        },
    ]
    last_output = ""
    for attempt in attempts:
        append_event(decisions, {"event": "full_attempt_start", **attempt})
        cmd = train_command(
            py,
            args,
            attempt["output_dir"],
            batch_size=attempt["batch_size"],
            grad_accum=attempt["grad_accum"],
            num_workers=4,
        )
        code, output = run_capture(cmd, decisions, tag=attempt["name"])
        last_output = output
        if code == 0:
            append_event(decisions, {"event": "full_attempt_success", **attempt})
            return attempt["output_dir"]
        if is_oom(output):
            append_event(decisions, {"event": "full_attempt_oom_fallback", **attempt, "tail": output[-4000:]})
            continue
        append_event(decisions, {"event": "full_attempt_non_oom_failure", **attempt, "tail": output[-4000:]})
        raise RuntimeError(f"CEAI full training failed for non-OOM reason in {attempt['name']}")
    raise RuntimeError(f"CEAI full training failed after all OOM fallbacks. Last output tail:\n{last_output[-4000:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground CEAI-OIA supervisor.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size1", type=int, default=3)
    ap.add_argument("--fallback_gradient_accumulation_steps1", type=int, default=11)
    ap.add_argument("--fallback_batch_size2", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps2", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bdd100k_root", default="E:/sbw/BDD100K")
    ap.add_argument("--bdd_oia_root", default="E:/sbw/BDD-OIA")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--goal_mode", action="store_true")
    args = ap.parse_args()
    preflight = Path(".background_runs/ceai_oia_v1_preflight")
    smoke = Path(".background_runs/ceai_oia_v1_smoke")
    full = Path(".background_runs/ceai_oia_v1_full_32")
    preflight.mkdir(parents=True, exist_ok=True)
    decisions = preflight / "supervisor_decisions.jsonl"
    append_event(
        decisions,
        {
            "event": "supervisor_start",
            "foreground": True,
            "no_feature_cache": bool(args.no_feature_cache),
            "test_only": bool(args.test_only),
            "goal_mode": bool(args.goal_mode),
            "requested_epochs": args.epochs,
        },
    )
    py = sys.executable
    compile_files = (
        list(Path("fate_oia/models").glob("ceai_*.py"))
        + list(Path("fate_oia/losses").glob("ceai_*.py"))
        + [
            Path("fate_oia/losses/gradient_budget.py"),
            Path("fate_oia/losses/pcgrad_lite.py"),
            Path("fate_oia/engine/train_ceai_oia.py"),
            Path("fate_oia/engine/audit_ceai_oia_implementation.py"),
            Path("fate_oia/engine/supervise_ceai_oia_foreground.py"),
            Path("fate_oia/datasets/bdd100k_scene_state.py"),
        ]
    )
    run([py, "-m", "py_compile", *[str(p) for p in compile_files]], decisions, tag="py_compile")
    run([py, "-m", "pytest", *[str(p) for p in Path("tests").glob("test_ceai_*.py")], "-q"], decisions, tag="pytest")
    run(
        [
            py,
            "-m",
            "fate_oia.engine.train_ceai_oia",
            "--config",
            args.config,
            "--output_dir",
            str(smoke),
            "--epochs",
            "1",
            "--batch_size",
            "2",
            "--gradient_accumulation_steps",
            "1",
            "--num_workers",
            "0",
            "--max_train_samples",
            "16",
            "--max_test_samples",
            "16",
            "--device",
            args.device,
            "--data_root",
            args.bdd_oia_root,
            "--raw_root",
            args.bdd_oia_root,
            "--bdd100k_root",
            args.bdd100k_root,
            "--best_selection_split",
            "test",
        ],
        decisions,
        tag="smoke",
    )
    run(
        [
            py,
            "-m",
            "fate_oia.engine.audit_ceai_oia_implementation",
            "--config",
            args.config,
            "--output_dir",
            str(preflight),
            "--smoke_dir",
            str(smoke),
            "--device",
            args.device,
        ],
        decisions,
        tag="audit",
    )
    pass_file = preflight / "REVIEW_PASS_CEAI_OIA_V1.txt"
    if args.require_review_pass and not pass_file.exists():
        raise RuntimeError("REVIEW_PASS_CEAI_OIA_V1.txt missing; refusing full training")
    completed_dir = run_full_with_oom_fallbacks(py, args, decisions, full)
    (completed_dir / "GOAL_COMPLETED_CEAI_OIA_V1.json").write_text(
        json.dumps({"status": "complete", "epochs": args.epochs, "output_dir": str(completed_dir)}, indent=2),
        encoding="utf-8",
    )
    append_event(decisions, {"event": "goal_completed", "output_dir": str(completed_dir), "epochs": args.epochs})


if __name__ == "__main__":
    main()
