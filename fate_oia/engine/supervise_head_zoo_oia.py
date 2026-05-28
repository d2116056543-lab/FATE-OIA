from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_C_REFERENCE = {"joint": 0.547844, "Exp_mF1": 0.381301, "Exp_mAP": 0.367822}
DEFAULT_HEAD_ORDER = [
    "h0_runc_compatible",
    "h4_runc_mrc_aux",
    "h1_q2l_decoder",
    "h2_ml_decoder_g8",
    "h3_ctran_masked",
    "h5_runc_calibrated",
]


@dataclass
class HeadZooDecision:
    continue_run: bool
    reason: str
    best_joint: float
    best_map: float
    best_epoch: int
    recent_improving: bool = False


def _metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _recent_gain(rows: list[dict[str, Any]], key: str, window: int = 3) -> float:
    if len(rows) < window:
        return 0.0
    tail = rows[-window:]
    return _metric(tail[-1], key) - _metric(tail[0], key)


def head_zoo_decision(
    rows: list[dict[str, Any]],
    *,
    min_gate_epoch: int = 14,
    max_epoch: int = 15,
    extension_epoch: int = 20,
    run_c_joint: float = RUN_C_REFERENCE["joint"],
    run_c_map: float = RUN_C_REFERENCE["Exp_mAP"],
) -> HeadZooDecision:
    if not rows:
        return HeadZooDecision(True, "no_metrics_yet", float("-inf"), float("-inf"), 0)
    best = max(rows, key=lambda r: _metric(r, "test_joint", float("-inf")))
    best_joint = _metric(best, "test_joint", float("-inf"))
    best_map = max(_metric(r, "test_Exp_mAP", float("-inf")) for r in rows)
    best_epoch = int(best.get("epoch", 0))
    last_epoch = max(int(r.get("epoch", 0)) for r in rows)
    if last_epoch < min_gate_epoch:
        return HeadZooDecision(True, "before_min_gate_epoch", best_joint, best_map, best_epoch)
    if any(not (float("-inf") < _metric(r, "test_joint", 0.0) < float("inf")) for r in rows[-1:]):
        return HeadZooDecision(False, "invalid_metric", best_joint, best_map, best_epoch)
    recent_map_gain = _recent_gain(rows, "test_Exp_mAP", 3)
    recent_exp_gain = _recent_gain(rows, "test_Exp_mF1", 3)
    recent_improving = (recent_map_gain >= 0.0015) or (recent_exp_gain >= 0.002)
    if best_joint < run_c_joint - 0.060 and best_map < run_c_map - 0.030 and not recent_improving:
        return HeadZooDecision(False, "severe_collapse_after_min_epoch", best_joint, best_map, best_epoch, recent_improving)
    if best_joint < run_c_joint - 0.030 and best_map < run_c_map - 0.010 and not recent_improving:
        return HeadZooDecision(False, "below_gate_after_min_epoch", best_joint, best_map, best_epoch, recent_improving)
    if last_epoch < max_epoch:
        return HeadZooDecision(True, "after_min_epoch_continue_to_nominal_max", best_joint, best_map, best_epoch, recent_improving)
    if last_epoch < extension_epoch and recent_improving and (best_joint >= run_c_joint - 0.015 or best_map >= run_c_map - 0.015):
        return HeadZooDecision(True, "near_runc_and_recently_improving", best_joint, best_map, best_epoch, recent_improving)
    return HeadZooDecision(False, "nominal_head_complete", best_joint, best_map, best_epoch, recent_improving)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _next_epoch(rows: list[dict[str, Any]]) -> int:
    return max([int(r.get("epoch", 0)) for r in rows] or [0]) + 1


def _stream_command(cmd: list[str], cwd: Path) -> int:
    print("HEADZOO_CMD " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return int(proc.wait())


def run_head(args: argparse.Namespace, head_name: str) -> dict[str, Any]:
    repo = Path(args.repo)
    head_dir = Path(args.run_root) / head_name
    head_dir.mkdir(parents=True, exist_ok=True)
    decisions = head_dir / "supervisor_decisions.jsonl"
    _append_jsonl(decisions, {"event": "head_supervisor_start", "head_name": head_name, "time": datetime.now().isoformat(), "min_gate_epoch": args.min_gate_epoch})
    batch_size = int(args.batch_size)
    accum = int(args.gradient_accumulation_steps)
    while True:
        rows = _read_metrics(head_dir / "metrics_summary.jsonl")
        epoch = _next_epoch(rows)
        if epoch > int(args.extension_epoch):
            break
        resume = head_dir / "checkpoint_latest.pth"
        cmd = [
            args.python,
            "-m",
            "fate_oia.engine.train_head_zoo_oia",
            "--output_dir",
            str(head_dir),
            "--head_name",
            head_name,
            "--data_root",
            args.data_root,
            "--raw_root",
            args.raw_root,
            "--run_c_dir",
            args.run_c_dir,
            "--pretrained_weights",
            args.pretrained_weights,
            "--epochs",
            str(epoch),
            "--batch_size",
            str(batch_size),
            "--gradient_accumulation_steps",
            str(accum),
            "--lr",
            str(args.lr),
            "--min_lr",
            str(args.min_lr),
            "--num_workers",
            str(args.num_workers),
            "--log_every",
            str(args.log_every),
        ]
        if resume.exists():
            cmd += ["--resume", str(resume)]
        _append_jsonl(decisions, {"event": "head_epoch_start", "head_name": head_name, "epoch": epoch, "cmd": cmd, "batch_size": batch_size, "accum": accum})
        rc = _stream_command(cmd, repo)
        if rc != 0 and batch_size == int(args.batch_size):
            _append_jsonl(decisions, {"event": "head_epoch_failed_retry_oom_fallback", "head_name": head_name, "epoch": epoch, "returncode": rc})
            batch_size = int(args.fallback_batch_size)
            accum = int(args.fallback_gradient_accumulation_steps)
            continue
        if rc != 0:
            _append_jsonl(decisions, {"event": "head_epoch_failed", "head_name": head_name, "epoch": epoch, "returncode": rc})
            return {"head_name": head_name, "status": "failed", "returncode": rc}
        rows = _read_metrics(head_dir / "metrics_summary.jsonl")
        decision = head_zoo_decision(rows, min_gate_epoch=args.min_gate_epoch, max_epoch=args.max_epoch, extension_epoch=args.extension_epoch, run_c_joint=args.run_c_joint, run_c_map=args.run_c_map)
        _append_jsonl(decisions, {"event": "head_decision", "head_name": head_name, **asdict(decision)})
        if not decision.continue_run:
            break
    rows = _read_metrics(head_dir / "metrics_summary.jsonl")
    best = max(rows, key=lambda r: _metric(r, "test_joint", float("-inf"))) if rows else {}
    summary = {"head_name": head_name, "status": "complete", "best": best, "rows": len(rows), "output_dir": str(head_dir)}
    (head_dir / "head_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground HeadZoo supervisor.")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--heads", nargs="+", default=DEFAULT_HEAD_ORDER)
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--run_c_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--fallback_batch_size", type=int, default=2)
    ap.add_argument("--fallback_gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=80)
    ap.add_argument("--min_gate_epoch", type=int, default=14)
    ap.add_argument("--max_epoch", type=int, default=15)
    ap.add_argument("--extension_epoch", type=int, default=20)
    ap.add_argument("--run_c_joint", type=float, default=RUN_C_REFERENCE["joint"])
    ap.add_argument("--run_c_map", type=float, default=RUN_C_REFERENCE["Exp_mAP"])
    args = ap.parse_args()
    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for head_name in args.heads:
        summaries[head_name] = run_head(args, head_name)
    (root / "head_zoo_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"event": "head_zoo_supervisor_complete", "run_root": str(root), "heads": list(summaries)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
