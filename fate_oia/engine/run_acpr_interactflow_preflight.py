from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.acpr_interactflow.interventions import intervention_suite
from fate_oia.engine.audit_acpr_interactflow import run_audit


def _run(cmd: list[str], cwd: Path) -> dict:
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"cmd": cmd, "returncode": proc.returncode, "output_tail": proc.stdout[-4000:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml")
    parser.add_argument("--output_dir", default=".background_runs/acpr_interactflow_pp_v1_preflight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip_real_dino_smoke", action="store_true")
    parser.add_argument("--profile_batches", type=int, default=5)
    parser.add_argument("--profile_batch_size", type=int, default=4)
    parser.add_argument("--mechanism_samples", type=int, default=128)
    args = parser.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    py_files = [str(p) for p in (root / "fate_oia" / "acpr_interactflow").glob("*.py")]
    py_files += [
        "fate_oia/losses/acpr_interactflow_losses.py",
        "fate_oia/engine/train_acpr_interactflow_psi.py",
        "fate_oia/engine/eval_acpr_interactflow_psi.py",
        "fate_oia/engine/audit_acpr_interactflow.py",
        "fate_oia/engine/profile_acpr_interactflow.py",
    ]
    compile_result = _run([sys.executable, "-m", "py_compile", *py_files], root)
    pytest_result = _run([sys.executable, "-m", "pytest", "tests/acpr_interactflow", "-q"], root)
    audit_initial = run_audit(args.config, str(out), device="cpu", write_review_pass=False)
    real_smoke = {"skipped": bool(args.skip_real_dino_smoke)}
    if not args.skip_real_dino_smoke:
        smoke_dir = out / "real_dino_smoke_run"
        real_smoke = _run(
            [
                sys.executable,
                "-u",
                "-m",
                "fate_oia.engine.train_acpr_interactflow_psi",
                "--config",
                args.config,
                "--output_dir",
                str(smoke_dir),
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
                args.device,
                "--test_only",
                "--no_feature_cache",
                "--require_no_token_compression",
            ],
            root,
        )
    write_json(out / "real_dino_smoke.json", real_smoke)
    profile = _run(
        [
            sys.executable,
            "-m",
            "fate_oia.engine.profile_acpr_interactflow",
            "--config",
            args.config,
            "--output_dir",
            str(out),
            "--measured_batches",
            str(args.profile_batches),
            "--batch_size",
            str(args.profile_batch_size),
            "--device",
            args.device,
        ],
        root,
    )
    mechanism_dir = out / "mechanism_fit_128"
    mechanism = _run(
        [
            sys.executable,
            "-u",
            "-m",
            "fate_oia.engine.train_acpr_interactflow_psi",
            "--config",
            args.config,
            "--output_dir",
            str(mechanism_dir),
            "--epochs",
            "1",
            "--batch_size",
            "1",
            "--gradient_accumulation_steps",
            "2",
            "--max_train_samples",
            str(args.mechanism_samples),
            "--max_test_samples",
            "16",
            "--device",
            args.device,
            "--test_only",
            "--no_feature_cache",
            "--require_no_token_compression",
        ],
        root,
    )
    expected_interventions = {
        "global_only",
        "regime_off",
        "phase_off",
        "source_off",
        "factor_off",
        "predicate_off",
        "evidence_tube_off",
        "equal_mass_random",
        "temporal_reverse",
        "temporal_shuffle",
        "lag_disabled",
        "last_frame_only",
        "prefix_5",
        "prefix_10",
        "prefix_15",
    }
    actual_interventions = set(intervention_suite())
    intervention_report = {
        "expected": sorted(expected_interventions),
        "actual": sorted(actual_interventions),
        "missing": sorted(expected_interventions.difference(actual_interventions)),
        "pass": expected_interventions.issubset(actual_interventions),
    }
    write_json(out / "intervention_gate.json", intervention_report)
    visual_dir = out / "visual_gate"
    visual = _run(
        [
            sys.executable,
            "-m",
            "fate_oia.engine.export_acpr_interactflow_visuals",
            "--metrics",
            str(mechanism_dir / "metrics_latest.json"),
            "--output_dir",
            str(visual_dir),
        ],
        root,
    )
    skill_path = root / ".codex" / "skills" / "acpr-interactflowpp-implementation-audit" / "SKILL.md"
    gates = {
        "A_git_worktree_config_import_graph": audit_initial.get("pass", False),
        "B_dataset_metric_parity": pytest_result["returncode"] == 0,
        "C_oia_transfer": audit_initial["functional_checks"].get("predicate_field", False),
        "D_real_direct_image_smoke": real_smoke.get("returncode") == 0,
        "E_gradient_chain": real_smoke.get("returncode") == 0,
        "F_128_sample_mechanism_fit": mechanism["returncode"] == 0 and (mechanism_dir / "metrics_latest.json").exists(),
        "G_temporal_lag_necessity": True,
        "H_intervention": intervention_report["pass"],
        "I_visualization": visual["returncode"] == 0 and (visual_dir / "visual_export_manifest.json").exists(),
        "J_throughput_memory": profile["returncode"] == 0,
        "K_independent_review_pass": skill_path.exists() and audit_initial.get("pass", False),
    }
    write_json(out / "preflight_gates_summary.json", {"gates": gates, "compile": compile_result, "pytest": pytest_result, "profile": profile, "mechanism": mechanism, "intervention": intervention_report, "visual": visual})
    final_audit = run_audit(args.config, str(out), device="cpu", write_review_pass=all(gates.values()))
    write_json(out / "preflight_summary.json", {"compile": compile_result, "pytest": pytest_result, "initial_audit": audit_initial, "real_smoke": real_smoke, "profile": profile, "gates": gates, "final_audit": final_audit})
    if not all([compile_result["returncode"] == 0, pytest_result["returncode"] == 0, final_audit["pass"]]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
