from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], decisions: Path) -> None:
    with decisions.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "run", "cmd": cmd}, ensure_ascii=False) + "\n")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"command failed code={code}: {' '.join(cmd)}")


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
    args = ap.parse_args()
    preflight = Path(".background_runs/ceai_oia_v1_preflight")
    smoke = Path(".background_runs/ceai_oia_v1_smoke")
    full = Path(".background_runs/ceai_oia_v1_full_32")
    preflight.mkdir(parents=True, exist_ok=True)
    decisions = preflight / "supervisor_decisions.jsonl"
    py = sys.executable
    run([py, "-m", "py_compile", *[str(p) for p in list(Path("fate_oia/models").glob("ceai_*.py")) + list(Path("fate_oia/losses").glob("ceai_*.py")) + [Path("fate_oia/losses/gradient_budget.py"), Path("fate_oia/losses/pcgrad_lite.py"), Path("fate_oia/engine/train_ceai_oia.py"), Path("fate_oia/engine/audit_ceai_oia_implementation.py"), Path("fate_oia/engine/supervise_ceai_oia_foreground.py"), Path("fate_oia/datasets/bdd100k_scene_state.py")]]], decisions)
    run([py, "-m", "pytest", *[str(p) for p in Path("tests").glob("test_ceai_*.py")], "-q"], decisions)
    run([py, "-m", "fate_oia.engine.train_ceai_oia", "--config", args.config, "--output_dir", str(smoke), "--epochs", "1", "--batch_size", "2", "--gradient_accumulation_steps", "1", "--num_workers", "0", "--max_train_samples", "16", "--max_test_samples", "16", "--device", args.device, "--data_root", args.bdd_oia_root, "--raw_root", args.bdd_oia_root, "--bdd100k_root", args.bdd100k_root], decisions)
    run([py, "-m", "fate_oia.engine.audit_ceai_oia_implementation", "--config", args.config, "--output_dir", str(preflight), "--smoke_dir", str(smoke), "--device", args.device], decisions)
    pass_file = preflight / "REVIEW_PASS_CEAI_OIA_V1.txt"
    if args.require_review_pass and not pass_file.exists():
        raise RuntimeError("REVIEW_PASS_CEAI_OIA_V1.txt missing; refusing full training")
    run([py, "-m", "fate_oia.engine.train_ceai_oia", "--config", args.config, "--output_dir", str(full), "--epochs", str(args.epochs), "--batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.gradient_accumulation_steps), "--num_workers", "4", "--max_train_samples", "0", "--max_test_samples", "0", "--device", args.device, "--data_root", args.bdd_oia_root, "--raw_root", args.bdd_oia_root, "--bdd100k_root", args.bdd100k_root], decisions)
    (full / "GOAL_COMPLETED_CEAI_OIA_V1.json").write_text(json.dumps({"status": "complete", "epochs": args.epochs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
