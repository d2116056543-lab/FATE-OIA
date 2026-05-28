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
class ScoreV2Decision:
    continue_stage: bool
    reason: str
    next_stage: str | None = None


def score_v2_stage1_decision(epoch: int, best_joint: float, best_exp_mf1: float, best_exp_map: float) -> ScoreV2Decision:
    if epoch >= 5 and best_joint < RUN_C_REFERENCE["joint"] - 0.020 and best_exp_map < RUN_C_REFERENCE["exp_map"] + 0.003:
        return ScoreV2Decision(False, "Stage1 stopped at epoch 5 gate: joint far below Run C and AP did not improve.", None)
    if epoch >= 8 and best_joint < RUN_C_REFERENCE["joint"] - 0.010 and best_exp_map < RUN_C_REFERENCE["exp_map"] + 0.005:
        return ScoreV2Decision(False, "Stage1 stopped at epoch 8 gate: no close trend to Run C.", None)
    if epoch >= 10 and best_exp_map > RUN_C_REFERENCE["exp_map"] + 0.010 and best_exp_mf1 <= RUN_C_REFERENCE["exp_mf1"]:
        return ScoreV2Decision(False, "Stage1 AP improved but F1 lags; route to calibration analysis.", "calibration_analysis")
    return ScoreV2Decision(True, "Stage1 continues.", None)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _run_child(cmd: list[str], output_dir: Path) -> int:
    print("[supervisor] running:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
    except KeyboardInterrupt:
        print("[supervisor] interrupted; terminating child", flush=True)
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
        except Exception:
            proc.terminate()
        _append_jsonl(output_dir / "supervisor_decisions.jsonl", {"event": "keyboard_interrupt", "time": datetime.now().isoformat()})
        raise
    return int(proc.wait())


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(rows, key=lambda r: (float(r.get("test_joint", -1e9)), float(r.get("test_Exp_mF1", -1e9))))


def _command(args: argparse.Namespace, output_dir: Path, *, batch_size: int, accum: int, epochs: int, smoke: bool) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_score_v2_oia",
        "--output_dir",
        str(output_dir),
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
        "--epochs",
        str(epochs),
        "--lr",
        str(args.lr),
        "--num_workers",
        str(args.num_workers),
        "--log_every",
        "1" if smoke else str(args.log_every),
        "--eval_splits",
        "test",
        "--stage",
        "frozen",
    ]
    if smoke:
        cmd += ["--max_train_samples", str(args.smoke_train_samples), "--max_test_samples", str(args.smoke_test_samples)]
    else:
        if args.stage1_max_train_samples > 0:
            cmd += ["--max_train_samples", str(args.stage1_max_train_samples)]
        if args.stage1_max_test_samples > 0:
            cmd += ["--max_test_samples", str(args.stage1_max_test_samples)]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground supervisor for ScoreV2 FATE-OIA.")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--smoke_train_samples", type=int, default=8)
    ap.add_argument("--smoke_test_samples", type=int, default=8)
    ap.add_argument("--stage1_max_train_samples", type=int, default=0)
    ap.add_argument("--stage1_max_test_samples", type=int, default=0)
    ap.add_argument("--allow_training", action="store_true")
    args = ap.parse_args()

    root = Path(args.output_dir) if args.output_dir else Path(".background_runs") / f"score_v2_foreground_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    root.mkdir(parents=True, exist_ok=True)
    _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "supervisor_started", "time": datetime.now().isoformat(), "run_c_reference": RUN_C_REFERENCE})
    smoke_dir = root / "smoke"
    code = _run_child(_command(args, smoke_dir, batch_size=1, accum=1, epochs=1, smoke=True), root)
    if code != 0:
        _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "smoke_failed", "returncode": code})
        raise SystemExit(code)
    _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "smoke_passed", "output_dir": str(smoke_dir)})
    if not args.allow_training:
        print("[supervisor] smoke complete; --allow_training not set, stopping before Stage1.", flush=True)
        return
    stage1_dir = root / "stage1_frozen"
    code = _run_child(
        _command(args, stage1_dir, batch_size=args.batch_size, accum=args.gradient_accumulation_steps, epochs=args.max_epochs, smoke=False),
        root,
    )
    if code != 0:
        # Basic OOM fallback: rerun once with same effective batch if the first attempt failed.
        _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "stage1_failed", "returncode": code, "fallback": "batch2_accum16"})
        fallback_dir = root / "stage1_frozen_fallback_b2"
        code = _run_child(
            _command(
                args,
                fallback_dir,
                batch_size=args.fallback_batch_size,
                accum=args.fallback_gradient_accumulation_steps,
                epochs=args.max_epochs,
                smoke=False,
            ),
            root,
        )
        stage1_dir = fallback_dir
    rows = _read_metrics(stage1_dir / "metrics_summary.jsonl")
    best = _best(rows)
    if not best:
        _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "no_metrics_after_stage1", "returncode": code})
        raise SystemExit(code or 1)
    decision = score_v2_stage1_decision(
        int(best.get("epoch", 0)),
        float(best.get("test_joint", -1.0)),
        float(best.get("test_Exp_mF1", -1.0)),
        float(best.get("test_Exp_mAP", -1.0)),
    )
    _append_jsonl(root / "supervisor_decisions.jsonl", {"event": "stage1_decision", "best": best, **asdict(decision)})
    print(json.dumps({"event": "score_v2_supervisor_complete", "best": best, "decision": asdict(decision)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
