from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_C_REFERENCE = {"joint": 0.547844, "act_mf1": 0.714387, "exp_mf1": 0.381301, "exp_map": 0.367822}


@dataclass
class Stage2Decision:
    continue_stage: bool
    reason: str


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda row: (float(row.get("test_joint", -1e9)), float(row.get("test_Exp_mF1", -1e9)))) if rows else {}


def score_v2_stage2_decision(
    *,
    epoch: int,
    rows: list[dict[str, Any]],
    reference_joint: float = RUN_C_REFERENCE["joint"],
    min_gate_epoch: int = 14,
    patience: int = 4,
    epsilon: float = 5e-4,
) -> Stage2Decision:
    if epoch < min_gate_epoch:
        return Stage2Decision(True, f"Stage2 warmup continues before epoch {min_gate_epoch}.")
    if not rows:
        return Stage2Decision(False, "Stage2 produced no metrics.")
    best = _best(rows)
    if float(best.get("test_joint", -1e9)) >= reference_joint:
        return Stage2Decision(True, "Stage2 has reached or exceeded Run C; continue observing until max epoch or plateau.")
    recent = rows[-patience:] if len(rows) >= patience else rows
    recent_best = max(float(row.get("test_joint", -1e9)) for row in recent)
    recent_start = float(recent[0].get("test_joint", -1e9))
    if len(recent) >= patience and recent_best <= recent_start + epsilon:
        return Stage2Decision(False, f"Stage2 plateau below Run C for {patience} epochs.")
    return Stage2Decision(True, "Stage2 is below Run C but still improving or not yet plateaued.")


def _run_child(cmd: list[str], output_dir: Path) -> int:
    print("[stage2-supervisor] running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("[stage2-supervisor] interrupted; terminating child", flush=True)
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except Exception:
            proc.terminate()
        _append_jsonl(output_dir / "supervisor_decisions.jsonl", {"event": "keyboard_interrupt", "time": datetime.now().isoformat()})
        raise
    return int(proc.wait())


def _base_args(args: argparse.Namespace, batch_size: int, accum: int, *, partial_init: bool) -> list[str]:
    cmd = [
        "--data_root",
        args.data_root,
        "--raw_root",
        args.raw_root,
        "--run_c_dir",
        args.run_c_dir,
        "--pretrained_weights",
        args.pretrained_weights,
        "--batch_size",
        str(batch_size),
        "--gradient_accumulation_steps",
        str(accum),
        "--lr",
        str(args.lr_head),
        "--lr_head",
        str(args.lr_head),
        "--lr_adapter",
        str(args.lr_adapter),
        "--num_workers",
        str(args.num_workers),
        "--log_every",
        str(args.log_every),
        "--eval_splits",
        "test",
        "--stage",
        "adaptformer",
        "--scheduler_total_epochs",
        str(args.max_epochs),
    ]
    if partial_init:
        cmd += ["--no-resume_optimizer", "--allow_partial_resume"]
    if int(getattr(args, "max_train_samples", 0)) > 0:
        cmd += ["--max_train_samples", str(args.max_train_samples)]
    if int(getattr(args, "max_test_samples", 0)) > 0:
        cmd += ["--max_test_samples", str(args.max_test_samples)]
    return cmd


def _epoch_command(args: argparse.Namespace, output_dir: Path, epoch: int, batch_size: int, accum: int) -> list[str]:
    latest = output_dir / "checkpoint_latest.pth"
    partial_init = not latest.exists()
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_score_v2_oia",
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(epoch),
        *_base_args(args, batch_size, accum, partial_init=partial_init),
    ]
    if latest.exists():
        cmd += ["--resume", str(latest)]
    else:
        cmd += ["--resume", str(args.stage1_checkpoint), "--resume_model_only"]
    return cmd


def run_stage2(args: argparse.Namespace, batch_size: int, accum: int) -> int:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _append_jsonl(out / "supervisor_decisions.jsonl", {"event": "stage2_supervisor_started", "time": datetime.now().isoformat(), "stage1_checkpoint": args.stage1_checkpoint, "run_c_reference": RUN_C_REFERENCE})
    for epoch in range(1, int(args.max_epochs) + 1):
        cmd = _epoch_command(args, out, epoch, batch_size, accum)
        _append_jsonl(out / "supervisor_decisions.jsonl", {"event": "stage2_epoch_start", "epoch": epoch, "cmd": cmd})
        code = _run_child(cmd, out)
        if code != 0:
            _append_jsonl(out / "supervisor_decisions.jsonl", {"event": "stage2_epoch_failed", "epoch": epoch, "returncode": code})
            return code
        rows = _read_metrics(out / "metrics_summary.jsonl")
        decision = score_v2_stage2_decision(epoch=epoch, rows=rows, reference_joint=args.reference_joint, min_gate_epoch=args.min_gate_epoch, patience=args.patience)
        payload = {"event": "stage2_epoch_decision", "epoch": epoch, "best": _best(rows), **asdict(decision)}
        _append_jsonl(out / "supervisor_decisions.jsonl", payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if not decision.continue_stage:
            return 0
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground Stage2 AdaptFormer supervisor for ScoreV2 FATE-OIA.")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stage1_checkpoint", required=True)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--min_gate_epoch", type=int, default=14)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--reference_joint", type=float, default=RUN_C_REFERENCE["joint"])
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--lr_adapter", type=float, default=2e-5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    args = ap.parse_args()
    code = run_stage2(args, args.batch_size, args.gradient_accumulation_steps)
    if code != 0:
        _append_jsonl(Path(args.output_dir) / "supervisor_decisions.jsonl", {"event": "stage2_retry_fallback", "batch_size": args.fallback_batch_size, "accum": args.fallback_gradient_accumulation_steps})
        code = run_stage2(args, args.fallback_batch_size, args.fallback_gradient_accumulation_steps)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
