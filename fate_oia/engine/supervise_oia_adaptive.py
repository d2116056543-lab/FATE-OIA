from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from fate_oia.engine.offline_fusion_alpha_sweep import run_alpha_sweep
from fate_oia.engine.offline_threshold_sweep import run_threshold_sweep
from fate_oia.engine.offline_per_label_failure_audit import run_failure_audit


BASELINE_JOINT = 0.547844
BASELINE_EXP_MF1 = 0.381301


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _append_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decide_next_branch(summary: dict[str, Any]) -> dict[str, Any]:
    threshold_gain = float(summary.get("threshold_gain_exp_mF1", 0.0))
    thresholded_joint_gain = float(summary.get("thresholded_joint_gain", 0.0))
    if threshold_gain >= 0.020 and thresholded_joint_gain >= 0.005:
        return {"recommended_next_run": "threshold_only", "rationale": "threshold/calibration gain is large enough before training"}
    if bool(summary.get("fusion_fix_recommended", False)):
        return {"recommended_next_run": "run_g_fusion_fix", "rationale": "alpha sweep shows visual logits add measurable value"}
    if bool(summary.get("long_tail_learning_problem", False)) or bool(summary.get("calibration_problem", False)):
        return {"recommended_next_run": "run_h_cooccur_longtail", "rationale": "reason labels/calibration dominate remaining error"}
    return {"recommended_next_run": "diagnostics_only", "rationale": "no branch is expected to improve without new design"}


def should_stop_run(
    completed_epoch_metrics: list[dict[str, Any]],
    baseline_joint: float = BASELINE_JOINT,
    baseline_exp_mF1: float = BASELINE_EXP_MF1,
) -> dict[str, Any]:
    if len(completed_epoch_metrics) < 2:
        return {"stop": False, "reason": "need_at_least_two_epochs"}
    recent = completed_epoch_metrics[-2:]
    if all(float(r.get("joint", 0.0)) < baseline_joint + 0.003 for r in recent):
        return {"stop": True, "reason": "no_improvement_two_epochs"}
    if all(float(r.get("Exp_mF1", 0.0)) < baseline_exp_mF1 - 0.012 for r in recent):
        return {"stop": True, "reason": "exp_mF1_drop_two_epochs"}
    if all(float(r.get("Act_mF1_fused", 1.0)) < 0.700 for r in recent) and not any(
        float(r.get("Exp_mF1", 0.0)) > baseline_exp_mF1 for r in recent
    ):
        return {"stop": True, "reason": "action_drop_without_exp_gain"}
    return {"stop": False, "reason": "continue"}


def _load_epoch_tensors(epoch_dir: Path) -> dict[str, torch.Tensor]:
    required = {
        "visual": "logits_action_visual_test.pt",
        "reason_action": "logits_action_reason_test.pt",
        "fused": "logits_action_fused_test.pt",
        "action_labels": "labels_action_test.pt",
        "reason_logits": "logits_reason_test.pt",
        "reason_labels": "labels_reason_test.pt",
    }
    missing = [name for name in required.values() if not (epoch_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing tensors in {epoch_dir}: {missing}")
    return {key: torch.load(epoch_dir / name, map_location="cpu") for key, name in required.items()}


def run_diagnostics(run_c_epoch_dir: Path, output_dir: Path) -> dict[str, Any]:
    tensors = _load_epoch_tensors(run_c_epoch_dir)
    alpha = run_alpha_sweep(
        visual_action_logits=tensors["visual"],
        reason_action_logits=tensors["reason_action"],
        labels_action=tensors["action_labels"],
        reason_logits=tensors["reason_logits"],
        labels_reason=tensors["reason_labels"],
        output_dir=output_dir,
        action_dim=int(tensors["action_labels"].shape[1]),
    )
    threshold = run_threshold_sweep(tensors["reason_logits"], tensors["reason_labels"], output_dir, prefix="Exp")
    audit = run_failure_audit(logits=tensors["reason_logits"], labels=tensors["reason_labels"], output_dir=output_dir)
    fixed_exp = float(threshold["fixed"]["metrics"].get("Exp_mF1", 0.0))
    per_label_exp = float(threshold["per_label"]["metrics"].get("Exp_mF1", 0.0))
    summary = {
        "best_alpha": alpha["best_alpha"],
        "fusion_alpha_gain": alpha["fusion_alpha_gain"],
        "fusion_fix_recommended": alpha["fusion_fix_recommended"],
        "threshold_gain_exp_mF1": per_label_exp - fixed_exp,
        "threshold_problem": (per_label_exp - fixed_exp) >= 0.020,
        "calibration_problem": (per_label_exp - fixed_exp) >= 0.010,
        "long_tail_learning_problem": any(row.get("group") == "tail" and float(row.get("AP", 0.0)) < 0.25 for row in audit["rows"]),
        "top_failed_reason_indices": audit["top_failed_reason_indices"],
    }
    decision = decide_next_branch({**summary, "thresholded_joint_gain": 0.0})
    summary.update(decision)
    (output_dir / "summary_next_action.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def ensure_label_cooccurrence(repo: Path, output_root: Path) -> Path:
    bias_path = repo / output_root / "fate_oia_train_label_cooccurrence.json"
    if bias_path.exists():
        return bias_path
    bias_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.build_label_cooccurrence",
        "--data_root",
        "dataset/BDD-OIA",
        "--raw_root",
        "raw_data/BDD-OIA",
        "--split",
        "train",
        "--action_dim",
        "4",
        "--reason_dim",
        "21",
        "--output",
        str(bias_path),
    ]
    subprocess.check_call(cmd, cwd=str(repo))
    return bias_path


def build_run_command(
    branch: str,
    repo: Path,
    output_dir: Path,
    checkpoint: Path,
    summary: dict[str, Any],
    *,
    label_bias_path: Path | None = None,
) -> list[str]:
    base = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_fate_oia",
        "--config",
        "configs/fate_oia_train_360x640.yaml",
        "--output_dir",
        str(output_dir),
        "--resume",
        str(checkpoint),
        "--no-resume_optimizer",
        "--resume_scheduler",
        "--no-resume_strict",
        "--pretrained_weights",
        "ckp/reference/dino_deitsmall8_pretrain.pth",
        "--epochs",
        "20",
        "--batch_size",
        "4",
        "--gradient_accumulation_steps",
        "8",
        "--lr",
        "0.00005",
        "--no-auto_scale_lr",
        "--scheduler",
        "cosine",
        "--min_lr",
        "0.00001",
        "--label_correlation",
        "self_attn",
        "--task_balance",
        "none",
        "--loss_counterfactual",
        "0",
        "--counterfactual_eval",
        "--token_compression",
        "keep_merge",
        "--compression_start_epoch",
        "0",
        "--compression_warmup_epochs",
        "0",
        "--compression_keep_ratio_start",
        "0.70",
        "--compression_keep_ratio_final",
        "0.70",
        "--num_summary_tokens",
        "4",
        "--min_tokens",
        "128",
        "--loss_grounding",
        "0.0001",
        "--grounding_mode",
        "both",
        "--best_selection_split",
        "test",
        "--best_selection_metric",
        "joint_test_score",
        "--device",
        "cuda",
        "--log_every",
        "1",
    ]
    if branch == "run_g_fusion_fix":
        alpha = str(summary.get("best_alpha", 0.1))
        base.extend(["--fusion_mode", "fixed_alpha", "--fusion_fixed_alpha", alpha])
    elif branch == "run_h_cooccur_longtail":
        if label_bias_path is None:
            raise ValueError("Run H requires a real label co-occurrence/PMI bias file.")
        base.extend(
            [
                "--fusion_mode",
                "reason_only",
                "--label_correlation_bias",
                "pmi",
                "--label_correlation_bias_path",
                str(label_bias_path),
                "--label_correlation_bias_weight",
                "0.05",
                "--reason_loss",
                "asl_logit_adjust",
                "--reason_prior_path",
                str(label_bias_path),
                "--reason_logit_adjust_tau",
                "0.2",
                "--reason_loss_weight",
                "1.25",
            ]
        )
    return base


def supervise_child(cmd: list[str], cwd: Path, events_path: Path) -> int:
    _write_jsonl(events_path, {"event": "child_start", "timestamp": _now(), "cmd": cmd})
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    last_status = time.monotonic()
    for line in proc.stdout:
        print(line, end="", flush=True)
        if '"event": "fate_oia_batch"' in line or '"event": "fate_oia_resume_loaded"' in line:
            try:
                _write_jsonl(events_path, json.loads(line))
            except Exception:
                pass
        if time.monotonic() - last_status > 60:
            _write_jsonl(events_path, {"event": "foreground_status", "timestamp": _now(), "child_pid": proc.pid})
            last_status = time.monotonic()
    code = proc.wait()
    _write_jsonl(events_path, {"event": "child_exit", "timestamp": _now(), "returncode": code})
    return int(code)


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground adaptive FATE-OIA supervisor.")
    ap.add_argument("--plan", default="adaptive_after_runC")
    ap.add_argument("--foreground", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--diagnostics_only", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--run_c_epoch_dir", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938/epoch_014")
    ap.add_argument("--checkpoint", default=".background_runs/fate_oia_runC_e13_cosine_labelcorr_20260526_191938/checkpoint_best_test.pth")
    ap.add_argument("--output_root", default=".background_runs")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    if not args.foreground:
        raise ValueError("This supervisor is foreground-only; do not disable --foreground.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = repo / args.output_root / f"adaptive_diagnostics_after_runC_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    events = output / "supervisor_events.jsonl"
    decision_trace = output / "decision_trace.jsonl"
    root = repo.parent
    _append_md(root / "progress.md", f"\n\n## {_now()} - Foreground adaptive supervisor started\n- Output: `{output}`\n- Checkpoint: `{args.checkpoint}`\n- Mode: diagnostics_only={args.diagnostics_only}\n")
    _write_jsonl(events, {"event": "supervisor_start", "timestamp": _now(), "output": str(output), "plan": args.plan})
    checkpoint = repo / args.checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    summary = run_diagnostics(repo / args.run_c_epoch_dir, output)
    _write_jsonl(decision_trace, {"event": "diagnostics_summary", "timestamp": _now(), **summary})
    _append_md(root / "findings.md", f"\n\n## {_now()} - Adaptive diagnostics summary\n- best_alpha={summary['best_alpha']}; fusion_alpha_gain={summary['fusion_alpha_gain']:.6f}; recommended_next_run={summary['recommended_next_run']}.\n- threshold_gain_exp_mF1={summary['threshold_gain_exp_mF1']:.6f}; top_failed_reason_indices={summary['top_failed_reason_indices']}.\n")
    if args.diagnostics_only or summary["recommended_next_run"] in {"threshold_only", "diagnostics_only"}:
        _write_jsonl(events, {"event": "supervisor_stop", "timestamp": _now(), "reason": summary["recommended_next_run"]})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    run_name = summary["recommended_next_run"]
    run_output = repo / args.output_root / f"fate_oia_{run_name}_{timestamp}"
    label_bias_path = ensure_label_cooccurrence(repo, repo / args.output_root) if run_name == "run_h_cooccur_longtail" else None
    if label_bias_path is not None:
        _write_jsonl(events, {"event": "label_bias_ready", "timestamp": _now(), "path": str(label_bias_path)})
    cmd = build_run_command(run_name, repo, run_output, checkpoint, summary, label_bias_path=label_bias_path)
    code = supervise_child(cmd, repo, events)
    (output / "experiment_summary.json").write_text(
        json.dumps({"diagnostics": summary, "run_output": str(run_output), "returncode": code}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
