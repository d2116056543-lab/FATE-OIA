from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_C_REFERENCE = {"joint": 0.547844, "Exp_mF1": 0.381301, "Exp_mAP": 0.367822}


def _metric(row: dict[str, Any], key: str, default: float = float("-inf")) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def choose_winner_head(results: dict[str, dict[str, Any]], *, run_c_joint: float, run_c_map: float, run_c_exp_mf1: float) -> dict[str, Any] | None:
    candidates = []
    for head_name, payload in results.items():
        best = payload.get("best") or {}
        joint = _metric(best, "test_joint")
        exp_map = _metric(best, "test_Exp_mAP")
        exp_mf1 = _metric(best, "test_Exp_mF1")
        reason = None
        if joint >= run_c_joint + 0.003:
            reason = "joint_gain"
        elif joint >= run_c_joint - 0.005 and exp_map >= run_c_map + 0.005:
            reason = "map_gain_with_joint_near_runc"
        elif exp_mf1 >= run_c_exp_mf1 + 0.005:
            reason = "exp_mf1_gain"
        if reason:
            candidates.append({"head_name": head_name, "winner_reason": reason, "joint": joint, "Exp_mAP": exp_map, "Exp_mF1": exp_mf1, "best": best})
    if not candidates:
        return None
    return max(candidates, key=lambda x: (x["joint"], x["Exp_mAP"], x["Exp_mF1"]))


def _run(cmd: list[str], cwd: Path) -> int:
    print("PIPELINE_CMD " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return int(proc.wait())


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_md(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + text.rstrip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Foreground FATE-OIA research pipeline supervisor.")
    ap.add_argument("--root", default="E:/sbw/FATE_Drive")
    ap.add_argument("--repo", default="E:/sbw/FATE_Drive/fate_oia_worktree")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--run_root", default="")
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--min_gate_epoch", type=int, default=14)
    ap.add_argument("--allow_training", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--best_split", default="test")
    ap.add_argument("--heads", nargs="+", default=["h0_runc_compatible", "h4_runc_mrc_aux", "h1_q2l_decoder", "h2_ml_decoder_g8", "h3_ctran_masked", "h5_runc_calibrated"])
    args = ap.parse_args()

    root = Path(args.root)
    repo = Path(args.repo)
    run_root = Path(args.run_root) if args.run_root else repo / ".background_runs" / f"oia_research_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)
    for md in ("task_plan.md", "findings.md", "progress.md"):
        if not (root / md).exists():
            raise FileNotFoundError(root / md)
    (run_root / "pipeline_manifest.json").write_text(json.dumps({"time": datetime.now().isoformat(), "args": vars(args), "min_gate_epoch": args.min_gate_epoch, "note": "Foreground pipeline; no background process."}, ensure_ascii=False, indent=2), encoding="utf-8")

    rc = _run([args.python, "-m", "fate_oia.engine.validate_baselines_oia", "--output_dir", str(run_root / "baseline_validation")], repo)
    if rc != 0:
        raise SystemExit(rc)
    if args.allow_training:
        rc = _run([
            args.python,
            "-m",
            "fate_oia.engine.supervise_head_zoo_oia",
            "--repo",
            str(repo),
            "--python",
            args.python,
            "--run_root",
            str(run_root / "head_zoo"),
            "--pretrained_weights",
            args.pretrained_weights,
            "--batch_size",
            str(args.batch_size),
            "--gradient_accumulation_steps",
            str(args.gradient_accumulation_steps),
            "--min_gate_epoch",
            str(args.min_gate_epoch),
            "--heads",
            *args.heads,
        ], repo)
        if rc != 0:
            raise SystemExit(rc)
    summary_path = run_root / "head_zoo" / "head_zoo_summary.json"
    summaries = _read_json(summary_path) if summary_path.exists() else {}
    winner = choose_winner_head(summaries, run_c_joint=RUN_C_REFERENCE["joint"], run_c_map=RUN_C_REFERENCE["Exp_mAP"], run_c_exp_mf1=RUN_C_REFERENCE["Exp_mF1"])
    decision = {"winner": winner, "run_c_reference": RUN_C_REFERENCE, "run_root": str(run_root)}
    (run_root / "pipeline_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _append_md(root / "progress.md", f"## {datetime.now().strftime('%Y-%m-%d %H:%M:%S %z')} - HeadZoo/PEFT/Evidence pipeline status\n\n- Run root: `{run_root}`\n- Min gate epoch: `{args.min_gate_epoch}`\n- Winner: `{winner}`\n- Boundary: PEFT/evidence phases are only allowed if a HeadZoo winner exists; this supervisor does not claim paper-level final results.\n")
    print(json.dumps({"event": "research_pipeline_complete", **decision}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
