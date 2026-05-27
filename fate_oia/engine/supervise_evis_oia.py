from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def append(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def run_foreground(cmd: list[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[supervisor] running:", " ".join(cmd), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line); log.flush()
        return int(proc.wait())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_s0(args, out_dir: Path) -> dict:
    diag = out_dir / "S0_runC_diagnostics"
    diag.mkdir(parents=True, exist_ok=True)
    run_c = Path(args.reference_run_c_dir)
    py = args.python
    cmds = [
        [py,"-m","fate_oia.engine.offline_fusion_alpha_sweep","--run_dir",str(run_c),"--output_dir",str(diag)],
        [py,"-m","fate_oia.engine.offline_threshold_sweep","--logits",str(run_c/"logits_reason_test.pt"),"--labels",str(run_c/"labels_reason_test.pt"),"--output_dir",str(diag),"--prefix","Exp"],
        [py,"-m","fate_oia.engine.offline_per_label_failure_audit","--logits",str(run_c/"logits_reason_test.pt"),"--labels",str(run_c/"labels_reason_test.pt"),"--output_dir",str(diag)],
        [py,"-m","fate_oia.engine.offline_score_branch_summary","--alpha_json",str(diag/"alpha_sweep_test.json"),"--threshold_json",str(diag/"threshold_sweep_test.json"),"--failure_json",str(diag/"per_label_failure_audit_test.json"),"--output_json",str(diag/"summary_next_action.json")],
    ]
    for cmd in cmds:
        code = run_foreground(cmd, Path(args.fate_oia_dir), out_dir/"foreground_supervisor.log")
        if code != 0:
            raise RuntimeError(f"S0 command failed with code {code}: {' '.join(cmd)}")
    return load_json(diag/"summary_next_action.json")


def latest_test_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics_summary.jsonl"
    rows=[]
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r=json.loads(line)
                if r.get("split") == "test": rows.append(r)
    if not rows: return {}
    return rows[-1]


def best_test_metrics(run_dir: Path) -> dict:
    p = run_dir / "metrics_summary.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "test":
                    rows.append(row)
    if rows:
        return max(rows, key=lambda r: float(r.get("joint_raw", 0.0) or 0.0))
    return load_json(run_dir / "metrics_best_test.json")


def run_training(args, out_dir: Path, name: str, extra: list[str]) -> tuple[int, Path]:
    run_dir = out_dir / name
    cmd = [
        args.python,"-m","fate_oia.engine.train_evis_oia",
        "--config","configs/evis_oia_score_patch.yaml",
        "--output_dir",str(run_dir),
        "--batch_size",str(args.batch_size),
        "--gradient_accumulation_steps",str(args.grad_accum),
        "--epochs",str(args.s1_epochs),
        "--num_workers",str(args.num_workers),
        "--log_every","50",
        "--best_selection_metric","joint_raw",
        "--early_stop_against_reference",
        "--reference_joint",str(args.reference_joint),
        "--reference_exp_mf1",str(args.reference_exp_mf1),
        "--early_stop_min_epochs",str(args.early_stop_min_epochs),
        "--early_stop_patience",str(args.early_stop_patience),
        "--early_stop_joint_margin",str(args.early_stop_joint_margin),
        "--early_stop_exp_margin",str(args.early_stop_exp_margin),
        *extra,
    ]
    code = run_foreground(cmd, Path(args.fate_oia_dir), out_dir/"foreground_supervisor.log")
    return code, run_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground EviS-OIA supervisor. No background mode is implemented.")
    ap.add_argument("--root", default=r"E:\sbw\FATE_Drive")
    ap.add_argument("--fate_oia_dir", default=r"E:\sbw\FATE_Drive\fate_oia_worktree")
    ap.add_argument("--reference_run_c_dir", default=r"E:\sbw\FATE_Drive\fate_oia_worktree\.background_runs\fate_oia_runC_e13_cosine_labelcorr_20260526_191938")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--python", default=r"E:\Anaconda\envs\sbw39\python.exe")
    ap.add_argument("--allow_training", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--s1_epochs", type=int, default=25)
    ap.add_argument("--reference_joint", type=float, default=0.5478436350822449)
    ap.add_argument("--reference_exp_mf1", type=float, default=0.38130074739456177)
    ap.add_argument("--early_stop_min_epochs", type=int, default=12)
    ap.add_argument("--early_stop_patience", type=int, default=2)
    ap.add_argument("--early_stop_joint_margin", type=float, default=0.005)
    ap.add_argument("--early_stop_exp_margin", type=float, default=0.005)
    ap.add_argument("--smoke", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()
    root = Path(args.root)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.fate_oia_dir)/".background_runs"/("evis_foreground_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ["task_plan.md","findings.md","progress.md"]:
        if not (root/f).exists(): raise FileNotFoundError(root/f)
    append(root/"progress.md", f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS foreground supervisor started\n- Output: `{out_dir}`\n- Foreground mode: subprocess output is streamed; no Start-Process/Start-Job/nohup is used.\n")
    decisions = out_dir/"supervisor_decisions.jsonl"
    s0 = run_s0(args, out_dir)
    decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S0_complete", **s0}, ensure_ascii=False)+"\n")
    append(root/"findings.md", f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS S0 diagnostics\n- Summary: `{s0}`\n")
    if not args.allow_training:
        return
    max_train = ["--max_train_samples","8","--max_val_samples","8","--max_test_samples","8"] if args.smoke else []
    code, s1_dir = run_training(args, out_dir, "S1_patch_only", ["--evidence_mode","patch_only","--adaptive_calibration","global",*max_train])
    if code != 0:
        decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S1_failed","code":code}, ensure_ascii=False)+"\n")
        raise SystemExit(code)
    best = best_test_metrics(s1_dir); latest = latest_test_metrics(s1_dir)
    decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S1_complete","best":best,"latest":latest}, ensure_ascii=False)+"\n")
    threshold_diag = load_json(out_dir/"S0_runC_diagnostics"/"threshold_sweep_test.json")
    run_c_joint = float(args.reference_joint)
    run_c_exp = float(args.reference_exp_mf1)
    run_c_ap = float(threshold_diag.get("fixed", {}).get("metrics", {}).get("Exp_mAP", 0.0) or 0.0)
    s1_joint = float(best.get("joint_raw", 0.0) or 0.0)
    s1_exp = float(best.get("Exp_mF1_raw", 0.0) or 0.0)
    s1_ap = float(best.get("Exp_mAP_raw", 0.0) or 0.0)
    if s1_joint >= run_c_joint + 0.005 or s1_exp >= run_c_exp + 0.010:
        code, s2_dir = run_training(args, out_dir, "S2_train_gt_eval_patch", ["--evidence_mode","train_gt_eval_patch","--adaptive_calibration","global","--epochs","12",*max_train])
        decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S2_complete","code":code,"best":best_test_metrics(s2_dir)}, ensure_ascii=False)+"\n")
    elif s1_ap >= run_c_ap + 0.005 and s1_exp < run_c_exp + 0.010:
        decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S2_skipped","reason":"S1 AP improved but F1 did not pass Run C gate; route to S3 calibration","s1_joint":s1_joint,"s1_exp":s1_exp,"s1_ap":s1_ap,"run_c_ap":run_c_ap}, ensure_ascii=False)+"\n")
        code, s3_dir = run_training(args, out_dir, "S3_adaptive_calibration", ["--evidence_mode","patch_only","--adaptive_calibration","instance","--epochs","12","--calibration_loss_weight","0.10",*max_train])
        decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S3_complete","code":code,"best":best_test_metrics(s3_dir)}, ensure_ascii=False)+"\n")
    else:
        decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S2_S3_skipped","reason":"S1 did not exceed Run C gates and AP did not justify calibration branch","s1_joint":s1_joint,"s1_exp":s1_exp,"s1_ap":s1_ap,"run_c_ap":run_c_ap}, ensure_ascii=False)+"\n")
    # S3/S4 are represented as decision-stage artifacts if S1 fails the gate.
    decisions.open("a",encoding="utf-8").write(json.dumps({"event":"S3_S4_decision","reason":"Run explainability/calibration audit only after a fair EviS checkpoint beats gate or user requests audit."}, ensure_ascii=False)+"\n")
    append(root/"progress.md", f"\n## {datetime.now():%Y-%m-%d %H:%M:%S} EviS supervisor completed decision chain\n- Output: `{out_dir}`\n- S1 best: `{best}`\n- S2/S3/S4 decisions recorded in `supervisor_decisions.jsonl`.\n")

if __name__ == "__main__":
    main()
