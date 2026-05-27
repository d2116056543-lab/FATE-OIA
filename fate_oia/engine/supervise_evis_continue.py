from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_foreground(cmd: list[str], cwd: Path, log_path: Path) -> tuple[int, str]:
    """Run a child process while keeping this supervisor in the foreground."""
    print("[continue-supervisor] running:", " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_chunks: list[str] = []
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            output_chunks.append(line)
        code = int(proc.wait())
    return code, "".join(output_chunks)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def best_test_metrics(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    p = run_dir / "metrics_summary.jsonl"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test":
                    rows.append(row)
    if rows:
        return max(rows, key=lambda r: float(r.get("joint_raw", 0.0) or 0.0))
    return load_json(run_dir / "metrics_best_test.json")


def latest_test_metrics(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "metrics_summary.jsonl"
    latest: dict[str, Any] = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test":
                    latest = row
    return latest


def metric_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def passes_main_gate(row: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        metric_float(row, "joint_raw") >= float(args.reference_joint) + float(args.required_joint_gain)
        or metric_float(row, "Exp_mF1_raw") >= float(args.reference_exp_mf1) + float(args.required_exp_gain)
    )


def ap_improved(row: dict[str, Any], run_c_ap: float, args: argparse.Namespace) -> bool:
    return metric_float(row, "Exp_mAP_raw") >= run_c_ap + float(args.required_ap_gain)


def train_stage(
    args: argparse.Namespace,
    out_dir: Path,
    stage_name: str,
    *,
    evidence_mode: str,
    adaptive_calibration: str,
    epochs: int,
    calibration_loss_weight: float,
    extra: list[str] | None = None,
) -> tuple[int, Path, dict[str, Any], dict[str, Any]]:
    extra = extra or []
    run_dir = out_dir / stage_name
    base_cmd = [
        args.python,
        "-m",
        "fate_oia.engine.train_evis_oia",
        "--config",
        "configs/evis_oia_score_patch.yaml",
        "--output_dir",
        str(run_dir),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation_steps",
        str(args.grad_accum),
        "--epochs",
        str(epochs),
        "--num_workers",
        str(args.num_workers),
        "--log_every",
        str(args.log_every),
        "--best_selection_metric",
        "joint_raw",
        "--early_stop_against_reference",
        "--reference_joint",
        str(args.reference_joint),
        "--reference_exp_mf1",
        str(args.reference_exp_mf1),
        "--early_stop_min_epochs",
        str(args.early_stop_min_epochs),
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--early_stop_joint_margin",
        str(args.early_stop_joint_margin),
        "--early_stop_exp_margin",
        str(args.early_stop_exp_margin),
        "--early_stop_required_joint_gain",
        str(args.required_joint_gain),
        "--early_stop_required_exp_gain",
        str(args.required_exp_gain),
        "--evidence_mode",
        evidence_mode,
        "--adaptive_calibration",
        adaptive_calibration,
        "--calibration_loss_weight",
        str(calibration_loss_weight),
        *extra,
    ]
    code, text = run_foreground(base_cmd, Path(args.fate_oia_dir), out_dir / "foreground_continue.log")
    oom = "out of memory" in text.lower() or "cuda error" in text.lower() and "memory" in text.lower()
    if code != 0 and oom and args.batch_size > 2:
        fallback_dir = out_dir / f"{stage_name}_b2_fallback"
        fallback_cmd = [x for x in base_cmd]
        fallback_cmd[fallback_cmd.index(str(run_dir))] = str(fallback_dir)
        fallback_cmd[fallback_cmd.index(str(args.batch_size))] = "2"
        fallback_cmd[fallback_cmd.index(str(args.grad_accum))] = "16"
        print("[continue-supervisor] OOM detected; retrying with batch=2, grad_accum=16", flush=True)
        code, _ = run_foreground(fallback_cmd, Path(args.fate_oia_dir), out_dir / "foreground_continue.log")
        run_dir = fallback_dir
    return code, run_dir, best_test_metrics(run_dir), latest_test_metrics(run_dir)


def logits_exist(run_dir: Path) -> bool:
    return (run_dir / "logits" / "logits_reason_raw_test.pt").exists() and (run_dir / "logits" / "labels_reason_test.pt").exists()


def run_s4_audit(args: argparse.Namespace, out_dir: Path, best_run_dir: Path, decisions: Path) -> dict[str, Any]:
    audit = out_dir / "S4_audit"
    audit.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "event": "S4_audit_started",
        "best_run_dir": str(best_run_dir),
        "threshold_audit": False,
        "failure_audit": False,
        "fate_snna_schema_smoke": False,
    }
    if logits_exist(best_run_dir):
        logits = best_run_dir / "logits" / "logits_reason_raw_test.pt"
        labels = best_run_dir / "logits" / "labels_reason_test.pt"
        cmds = [
            [
                args.python,
                "-m",
                "fate_oia.engine.offline_threshold_sweep",
                "--logits",
                str(logits),
                "--labels",
                str(labels),
                "--output_dir",
                str(audit),
                "--prefix",
                "Exp_raw",
            ],
            [
                args.python,
                "-m",
                "fate_oia.engine.offline_per_label_failure_audit",
                "--logits",
                str(logits),
                "--labels",
                str(labels),
                "--output_dir",
                str(audit),
            ],
        ]
        for cmd in cmds:
            code, _ = run_foreground(cmd, Path(args.fate_oia_dir), out_dir / "foreground_continue.log")
            result["threshold_audit" if "offline_threshold_sweep" in cmd else "failure_audit"] = code == 0
    else:
        result["logits_missing"] = True
    visual_cmd = [
        args.python,
        "-m",
        "fate_oia.engine.export_fate_snna_visuals",
        "--checkpoint",
        str(best_run_dir / "checkpoint_best_test.pth"),
        "--output_dir",
        str(audit / "fate_snna_schema"),
        "--max_samples",
        "2",
        "--label_indices",
        "0,4,9",
        "--methods",
        "label_attention,grad_x_attention",
    ]
    code, _ = run_foreground(visual_cmd, Path(args.fate_oia_dir), out_dir / "foreground_continue.log")
    result["fate_snna_schema_smoke"] = code == 0
    result["boundary"] = "S4 exports offline threshold/failure audits and FATE-SNNA schema smoke. It is not a trained checkpoint-dependent grounding/deletion proof."
    write_json(audit / "s4_summary.json", result)
    append_jsonl(decisions, result)
    return result


def choose_best_run(candidates: list[tuple[str, Path, dict[str, Any]]]) -> tuple[str, Path, dict[str, Any]]:
    valid = [(name, path, row) for name, path, row in candidates if row]
    if not valid:
        raise RuntimeError("No valid candidate run metrics available for S4 audit.")
    return max(valid, key=lambda item: metric_float(item[2], "joint_raw"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Continue EviS-OIA foreground supervisor from existing S1 into bounded S2-S4.")
    ap.add_argument("--root", default=r"E:\sbw\FATE_Drive")
    ap.add_argument("--fate_oia_dir", default=r"E:\sbw\FATE_Drive\fate_oia_worktree")
    ap.add_argument("--existing_s1_dir", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\evis_foreground_20260527_152220\S1_patch_only")
    ap.add_argument("--s0_diag_dir", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\evis_s0_diagnostics_20260527_151154\S0_runC_diagnostics")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--python", default=r"E:\Anaconda\envs\sbw39\python.exe")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--s2_epochs", type=int, default=12)
    ap.add_argument("--s3_epochs", type=int, default=12)
    ap.add_argument("--early_stop_min_epochs", type=int, default=8)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--early_stop_joint_margin", type=float, default=0.005)
    ap.add_argument("--early_stop_exp_margin", type=float, default=0.005)
    ap.add_argument("--required_joint_gain", type=float, default=0.005)
    ap.add_argument("--required_exp_gain", type=float, default=0.010)
    ap.add_argument("--required_ap_gain", type=float, default=0.005)
    ap.add_argument("--reference_joint", type=float, default=0.5478436350822449)
    ap.add_argument("--reference_exp_mf1", type=float, default=0.38130074739456177)
    ap.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    root = Path(args.root)
    for name in ["task_plan.md", "findings.md", "progress.md"]:
        if not (root / name).exists():
            raise FileNotFoundError(root / name)

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.fate_oia_dir) / ".background_runs" / ("evis_continue_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions = out_dir / "supervisor_decisions.jsonl"
    append_text(
        root / "progress.md",
        f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS continuation supervisor started\n"
        f"- Output: `{out_dir}`\n"
        "- Mode: foreground streaming continuation from existing S1 into bounded S2/S3/S4.\n"
        f"- Existing S1: `{args.existing_s1_dir}`\n",
    )

    existing_s1 = Path(args.existing_s1_dir)
    s1_best = best_test_metrics(existing_s1)
    s1_latest = latest_test_metrics(existing_s1)
    threshold_diag = load_json(Path(args.s0_diag_dir) / "threshold_sweep_test.json")
    run_c_ap = float(threshold_diag.get("fixed", {}).get("metrics", {}).get("Exp_mAP", 0.0) or 0.0)
    append_jsonl(decisions, {"event": "existing_S1_loaded", "best": s1_best, "latest": s1_latest, "run_c_ap": run_c_ap})

    max_samples: list[str] = []
    if args.smoke:
        max_samples = ["--max_train_samples", "8", "--max_val_samples", "8", "--max_test_samples", "8"]

    candidates: list[tuple[str, Path, dict[str, Any]]] = [("S1_existing", existing_s1, s1_best)]

    append_jsonl(decisions, {"event": "S2_forced_by_user", "reason": "User requested S2-S4 continuation; S2 remains bounded by early-stop gates even though S1 did not beat Run C."})
    code2, s2_dir, s2_best, s2_latest = train_stage(
        args,
        out_dir,
        "S2_train_gt_eval_patch",
        evidence_mode="train_gt_eval_patch",
        adaptive_calibration="global",
        epochs=args.s2_epochs,
        calibration_loss_weight=0.05,
        extra=max_samples,
    )
    append_jsonl(decisions, {"event": "S2_complete", "code": code2, "best": s2_best, "latest": s2_latest})
    candidates.append(("S2_train_gt_eval_patch", s2_dir, s2_best))
    if code2 != 0:
        append_jsonl(decisions, {"event": "S2_nonzero_exit", "code": code2, "action": "Proceed to S4 audit on best available checkpoint; do not launch S3."})
    else:
        s3_condition = passes_main_gate(s2_best, args) or ap_improved(s2_best, run_c_ap, args) or ap_improved(s1_best, run_c_ap, args)
        append_jsonl(
            decisions,
            {
                "event": "S3_gate",
                "run": "S2_train_gt_eval_patch",
                "passes_main_gate": passes_main_gate(s2_best, args),
                "s2_ap_improved": ap_improved(s2_best, run_c_ap, args),
                "s1_ap_improved": ap_improved(s1_best, run_c_ap, args),
                "decision": "run_S3" if s3_condition else "skip_S3",
            },
        )
        if s3_condition:
            code3, s3_dir, s3_best, s3_latest = train_stage(
                args,
                out_dir,
                "S3_adaptive_calibration",
                evidence_mode="train_gt_eval_patch",
                adaptive_calibration="instance",
                epochs=args.s3_epochs,
                calibration_loss_weight=0.10,
                extra=max_samples,
            )
            append_jsonl(decisions, {"event": "S3_complete", "code": code3, "best": s3_best, "latest": s3_latest})
            candidates.append(("S3_adaptive_calibration", s3_dir, s3_best))
        else:
            append_jsonl(decisions, {"event": "S3_skipped", "reason": "Neither S1 nor S2 showed enough AP/main-gate signal for calibration branch."})

    best_name, best_dir, best_row = choose_best_run(candidates)
    append_jsonl(decisions, {"event": "best_available_for_S4", "name": best_name, "dir": str(best_dir), "best": best_row})
    s4 = run_s4_audit(args, out_dir, best_dir, decisions)
    append_text(
        root / "progress.md",
        f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS continuation supervisor completed S2-S4 decision chain\n"
        f"- Output: `{out_dir}`\n"
        f"- Best available EviS run for S4: `{best_name}` at `{best_dir}`\n"
        f"- Best metrics: `{best_row}`\n"
        f"- S4 summary: `{s4}`\n",
    )
    append_text(
        root / "findings.md",
        f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS S2-S4 continuation finding\n"
        f"- Best available EviS candidate after bounded continuation: `{best_name}`.\n"
        f"- Metrics: `{best_row}`\n"
        "- Boundary: S4 audit here is offline threshold/failure/FATE-SNNA schema audit, not final checkpoint-dependent visual faithfulness proof.\n",
    )
    print(json.dumps({"event": "evis_continue_complete", "output_dir": str(out_dir), "best_name": best_name, "best": best_row, "s4": s4}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
