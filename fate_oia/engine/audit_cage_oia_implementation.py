from __future__ import annotations

import argparse
import json
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


REQUIRED_FILES = [
    "fate_oia/models/cage_label_nodes.py",
    "fate_oia/models/cage_evidence_retriever.py",
    "fate_oia/models/cage_dynamic_transport.py",
    "fate_oia/models/cage_reason_reliability.py",
    "fate_oia/models/cage_oia_model.py",
    "fate_oia/losses/cage_losses.py",
    "fate_oia/engine/train_cage_oia.py",
    "configs/cage_oia_v1_360x640.yaml",
]


COMPILE_FILES = [
    "fate_oia/models/cage_label_nodes.py",
    "fate_oia/models/cage_evidence_retriever.py",
    "fate_oia/models/cage_dynamic_transport.py",
    "fate_oia/models/cage_reason_reliability.py",
    "fate_oia/models/cage_oia_model.py",
    "fate_oia/losses/cage_losses.py",
    "fate_oia/engine/train_cage_oia.py",
]


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=".background_runs/cage_oia_v1_preflight")
    parser.add_argument("--real_smoke_dir", default="", help="Optional direct-image smoke dir to hard-check for full trainer artifacts.")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for rel in REQUIRED_FILES:
        if not (ROOT / rel).exists():
            failures.append(f"missing required file: {rel}")

    for rel in COMPILE_FILES:
        try:
            py_compile.compile(str(ROOT / rel), doraise=True)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"py_compile failed for {rel}: {exc}")

    pytest_cmd = [sys.executable, "-m", "pytest", "tests/test_cage_label_nodes.py", "tests/test_cage_evidence_transport.py", "tests/test_cage_full_trainer_contract.py", "-q"]
    pytest_result = run(pytest_cmd, ROOT)
    (out / "pytest_output.txt").write_text(pytest_result.stdout, encoding="utf-8")
    if pytest_result.returncode != 0:
        failures.append("targeted pytest failed")

    smoke_dir = out / "smoke"
    smoke_cmd = [sys.executable, "-m", "fate_oia.engine.train_cage_oia", "--smoke_only", "--output_dir", str(smoke_dir)]
    smoke_result = run(smoke_cmd, ROOT)
    (out / "smoke_output.txt").write_text(smoke_result.stdout, encoding="utf-8")
    if smoke_result.returncode != 0:
        failures.append("smoke failed")

    summary_path = smoke_dir / "cage_smoke_summary.json"
    selected_path = smoke_dir / "selected_vs_random_by_label.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    if not summary:
        failures.append("missing cage_smoke_summary.json")
    if not selected:
        failures.append("missing selected_vs_random_by_label.json")
    if summary.get("evidence_state_shape", [None, None])[1] != 25:
        failures.append("evidence is not label-specific with 25 labels")
    typed = summary.get("typed_edge_shapes", {})
    for key in ["A_A", "A_R", "R_A", "R_R"]:
        if key not in typed:
            failures.append(f"missing typed edge {key}")
    if summary.get("reason_reliability_shape", [None, None])[1] != 21:
        failures.append("reason reliability is not per reason label")
    if summary.get("action_gate_min", 1.0) > 0.35:
        failures.append("branch-safe action gate should start low")
    if summary.get("test_forward_uses_bdd100k_gt") is not False:
        failures.append("test-forward leakage assertion failed")
    if len(selected.get("per_label", [])) != 25:
        failures.append("selected-vs-random schema must contain 25 labels")
    if selected.get("available") is not False:
        failures.append("smoke selected-vs-random must not pretend real deletion is available")


    trainer_text = (ROOT / "fate_oia/engine/train_cage_oia.py").read_text(encoding="utf-8")
    if "raise NotImplementedError" in trainer_text:
        failures.append("train_cage_oia still contains a full-training NotImplementedError")
    if "build_backbone" not in trainer_text or "BDDOIAMultiTaskDataset" not in trainer_text:
        failures.append("train_cage_oia does not use real direct-image BDD-OIA/DINO path")
    if "selected_vs_random_action_drop" not in trainer_text or "action_gt_loss_drop" not in trainer_text:
        failures.append("train_cage_oia does not implement real selected-vs-random action-GT loss drop")

    if args.real_smoke_dir:
        real_dir = Path(args.real_smoke_dir)
        if not real_dir.is_absolute():
            real_dir = ROOT / real_dir
        required_real = [
            real_dir / "run_manifest.json",
            real_dir / "metrics_summary.jsonl",
            real_dir / "checkpoint_best_test.pth",
            real_dir / "selected_vs_random_by_label.jsonl",
        ]
        for path in required_real:
            if not path.exists():
                failures.append(f"missing real direct-image smoke artifact: {path}")
        if (real_dir / "selected_vs_random_by_label.jsonl").exists():
            rows = [json.loads(line) for line in (real_dir / "selected_vs_random_by_label.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows or not all(row.get("available") is True and row.get("judge") == "action_gt_loss_drop" for row in rows):
                failures.append("real smoke selected-vs-random artifact is not true action_gt_loss_drop")
        if (real_dir / "run_manifest.json").exists():
            manifest = json.loads((real_dir / "run_manifest.json").read_text(encoding="utf-8"))
            if manifest.get("eval_splits") != ["test"] or manifest.get("best_selection_split") != "test":
                failures.append("real smoke manifest is not test-only / test-best")

    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "summary": summary, "selected_schema": selected}
    (out / "audit_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps(result, indent=2))
        raise SystemExit(1)
    pass_file = out / "REVIEW_PASS_CAGE_OIA_V1.txt"
    pass_file.write_text("PASS\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
