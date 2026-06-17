from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PY_COMPILE_FILES = [
    "fate_oia/models/acpr_predicate_patch_targets.py",
    "fate_oia/models/acpr_predicate_transport_alignment.py",
    "fate_oia/models/acpr_triadic_mediator.py",
    "fate_oia/models/acpr_predicate_conditioned_threshold.py",
    "fate_oia/models/acpr_oia_model.py",
    "fate_oia/models/acpr_scene_predicate_head.py",
    "fate_oia/models/acpr_predicate_targets.py",
    "fate_oia/models/acpr_predicate_reason.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_threshold_head.py",
    "fate_oia/models/acpr_pair_memory.py",
    "fate_oia/losses/acpr_pmt_losses.py",
    "fate_oia/engine/train_acpr_oia.py",
    "fate_oia/engine/eval_acpr_oia.py",
    "fate_oia/engine/audit_acpr_pmt_s_implementation.py",
    "fate_oia/engine/export_acpr_pmt_visuals.py",
    "fate_oia/engine/supervise_acpr_pmt_s_foreground.py",
]

PYTEST_FILES = [
    "tests/test_acpr_predicate_patch_targets.py",
    "tests/test_acpr_predicate_transport_alignment.py",
    "tests/test_acpr_triadic_mediator.py",
    "tests/test_acpr_predicate_conditioned_threshold.py",
    "tests/test_acpr_pmt_forward_equivalence.py",
    "tests/test_acpr_pmt_losses.py",
    "tests/test_acpr_pmt_pair_filtering.py",
    "tests/test_acpr_pmt_phase_schedule.py",
    "tests/test_acpr_pmt_artifacts.py",
    "tests/test_acpr_pmt_audit.py",
    "tests/test_acpr_pmt_supervisor.py",
]


def stream_run(cmd: list[str]) -> tuple[int, str]:
    print("[pmt_supervisor]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if len(tail) > 200:
            tail = tail[-200:]
    return int(proc.wait()), "".join(tail)


def require_success(cmd: list[str]) -> None:
    code, tail = stream_run(cmd)
    if code != 0:
        raise SystemExit(f"preflight command failed ({code}): {' '.join(cmd)}\n{tail[-2000:]}")


def load_fallback_ladder(config_path: str, first_batch: int, first_accum: int) -> list[tuple[int, int]]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    ladder = cfg.get("training", {}).get("fallback_ladder") or []
    parsed = [(int(x[0]), int(x[1])) for x in ladder if isinstance(x, list) and len(x) == 2]
    if (first_batch, first_accum) not in parsed:
        parsed.insert(0, (first_batch, first_accum))
    return parsed


def run_preflight(py: str, config: str) -> None:
    for p in [r"E:\sbw\FATE_Drive\task_plan.md", r"E:\sbw\FATE_Drive\findings.md", r"E:\sbw\FATE_Drive\progress.md"]:
        Path(p).read_text(encoding="utf-8", errors="ignore")
    require_success([py, "-m", "py_compile", *PY_COMPILE_FILES])
    require_success([py, "-m", "pytest", *PYTEST_FILES, "-q"])
    require_success([
        py,
        "-m",
        "fate_oia.engine.audit_acpr_pmt_s_implementation",
        "--config",
        config,
        "--output_dir",
        ".background_runs/acpr_pmt_s_v1_preflight",
        "--device",
        "cuda",
        "--write_review_pass",
    ])
    review_pass = Path(".background_runs/acpr_pmt_s_v1_preflight/REVIEW_PASS_ACPR_PMT_S_V1.txt")
    if not review_pass.exists():
        raise SystemExit("missing REVIEW_PASS_ACPR_PMT_S_V1.txt after audit")
    require_success([
        py,
        "-u",
        "-m",
        "fate_oia.engine.train_acpr_oia",
        "--config",
        config,
        "--output_dir",
        ".background_runs/acpr_pmt_s_v1_smoke",
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
        "--device",
        "cuda",
        "--test_only",
        "--no_feature_cache",
        "--require_no_token_compression",
    ])
    require_success(["git", "add", "configs", "fate_oia", "tests", "scripts", ".codex"])
    code, staged = stream_run(["git", "diff", "--cached", "--quiet"])
    if code != 0:
        require_success(["git", "commit", "-m", "Complete ACPR-PMT-S V1 preflight functionality"])
    require_success(["git", "push", "github", "HEAD:acpr_pmt_s_v1"])
    code, local = stream_run(["git", "rev-parse", "HEAD"])
    if code != 0:
        raise SystemExit("failed to read local HEAD")
    code, remote = stream_run(["git", "ls-remote", "github", "refs/heads/acpr_pmt_s_v1"])
    if code != 0 or local.strip() not in remote:
        raise SystemExit(f"remote HEAD verification failed: local={local.strip()} remote={remote.strip()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--gradient_accumulation_steps", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--require_review_pass", action="store_true")
    args = ap.parse_args()
    py = sys.executable
    if args.require_review_pass:
        run_preflight(py, args.config)
    else:
        for p in [r"E:\sbw\FATE_Drive\task_plan.md", r"E:\sbw\FATE_Drive\findings.md", r"E:\sbw\FATE_Drive\progress.md"]:
            Path(p).read_text(encoding="utf-8", errors="ignore")
    attempts = load_fallback_ladder(args.config, args.batch_size, args.gradient_accumulation_steps)
    last_tail = ""
    for i, (batch, accum) in enumerate(attempts):
        cmd = [
            py, "-u", "-m", "fate_oia.engine.train_acpr_oia",
            "--config", args.config,
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(batch),
            "--gradient_accumulation_steps", str(accum),
            "--device", args.device,
            "--test_only",
            "--no_feature_cache",
            "--require_no_token_compression",
        ]
        code, tail = stream_run(cmd)
        if code == 0:
            return
        last_tail = tail
        oom = "out of memory" in tail.lower() or "cuda oom" in tail.lower()
        if not oom or i == len(attempts) - 1:
            raise SystemExit(code)
        print(f"[pmt_supervisor] CUDA OOM detected; retrying fallback batch={attempts[i+1][0]} accum={attempts[i+1][1]}", flush=True)
    raise SystemExit(f"training failed after fallbacks: {last_tail[-1000:]}")


if __name__ == "__main__":
    main()
