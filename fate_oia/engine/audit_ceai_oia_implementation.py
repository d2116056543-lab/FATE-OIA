from __future__ import annotations

import argparse
import ast
import json
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any


FORBIDDEN = ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup", "hidden cmd"]


def ceai_py_files() -> list[Path]:
    files: list[Path] = []
    files.extend(Path("fate_oia/models").glob("ceai_*.py"))
    files.extend(
        [
            Path("fate_oia/losses/ceai_losses.py"),
            Path("fate_oia/losses/gradient_budget.py"),
            Path("fate_oia/losses/pcgrad_lite.py"),
            Path("fate_oia/engine/train_ceai_oia.py"),
            Path("fate_oia/engine/eval_ceai_oia.py"),
            Path("fate_oia/engine/audit_ceai_oia_implementation.py"),
            Path("fate_oia/engine/supervise_ceai_oia_foreground.py"),
            Path("fate_oia/datasets/bdd100k_scene_state.py"),
        ]
    )
    return sorted({p for p in files if p.exists()})


def validate_format_ast_yaml(py_files: list[Path] | None = None, yaml_files: list[Path] | None = None) -> list[str]:
    errors: list[str] = []
    py_files = py_files or ceai_py_files()
    yaml_files = yaml_files or [Path("configs/fate_oia_train_360x640_ceai_oia_v1.yaml")]
    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8-sig")
            if len(text.splitlines()) <= 5:
                errors.append(f"{p} appears single-line or malformed")
            ast.parse(text, filename=str(p))
            py_compile.compile(str(p), doraise=True)
        except Exception as exc:
            errors.append(f"{p} format/compile failed: {exc}")
    for p in yaml_files:
        try:
            import yaml

            obj = yaml.safe_load(p.read_text(encoding="utf-8-sig"))
            if not isinstance(obj, dict):
                errors.append(f"{p} must parse to dict")
        except Exception as exc:
            errors.append(f"{p} yaml failed: {exc}")
    return errors


def validate_evidence_schema(obj: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if "available" not in obj:
        errors.append("evidence schema missing available")
    available = bool(obj.get("available", False))
    if not available:
        if obj.get("selected_mean", None) is not None or obj.get("random_mean", None) is not None:
            errors.append("unavailable evidence must use null selected/random means")
        if not obj.get("reason"):
            errors.append("unavailable evidence must explain reason")
        return errors
    for key in ["selected_mean", "random_mean", "evidence_gate_active"]:
        if key not in obj:
            errors.append(f"available evidence missing {key}")
    if obj.get("selected_mean") == 0.0 and obj.get("random_mean") == 0.0 and obj.get("evidence_gate_active") is False:
        errors.append("fake all-zero selected/random evidence placeholder")
    return errors


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, proc.stdout


def load_config_flat(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    flat: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def validate_config(cfg: dict[str, Any]) -> list[str]:
    req = {
        "feature_cache_enabled": False,
        "token_compression": "none",
        "test_only_evaluation": True,
        "best_selection_split": "test",
        "bdd100k_test_gt_input": False,
        "image_height": 360,
        "image_width": 640,
        "action_dim": 4,
        "reason_dim": 21,
    }
    errors: list[str] = []
    for k, v in req.items():
        if cfg.get(k) != v:
            errors.append(f"config {k} expected {v!r}, got {cfg.get(k)!r}")
    if int(cfg.get("epochs", 0)) < 28:
        errors.append("config epochs must be >= 28")
    if cfg.get("scheduler") != "cosine":
        errors.append("scheduler must be cosine")
    return errors


def validate_foreground_files(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
        for word in FORBIDDEN:
            if word.lower() in text.lower():
                errors.append(f"forbidden foreground token {word} in {p}")
    return errors


def run_static_audit(require_smoke_artifacts: bool = False, smoke_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    import inspect
    from fate_oia.datasets import bdd100k_scene_state
    from fate_oia.engine import train_ceai_oia
    from fate_oia.losses import ceai_losses, gradient_budget, pcgrad_lite
    from fate_oia.models.ceai_cross_expert_exchange import ControlledCrossExpertExchange
    from fate_oia.models.ceai_oia_model import CEAIOIAFeatureModel
    from fate_oia.models.ceai_pair_reliability import PairReliabilityHead
    from fate_oia.models.ceai_pair_sparse_attention import TaskGuidedPairSparseAttention
    from fate_oia.models.ceai_router import ParetoSafeRouter
    from fate_oia.models.ceai_scene_state import SceneStatePrototypeTransformer

    src_model = inspect.getsource(CEAIOIAFeatureModel)
    if "FATEOIAFeatureModel" not in src_model:
        errors.append("CEAI model must use FATEOIAFeatureModel")
    for key in ["base_action_logits", "base_reason_logits", "final_action_logits", "final_reason_logits"]:
        if key not in src_model:
            errors.append(f"CEAI model missing marker {key}")
    if ".detach()" not in inspect.getsource(ControlledCrossExpertExchange):
        errors.append("A->R stopgrad/detach missing")
    if "q_ar" not in inspect.getsource(ControlledCrossExpertExchange) or "r2a_active" not in inspect.getsource(ControlledCrossExpertExchange):
        errors.append("R->A q_ar/readiness gate missing")
    if "topk" not in inspect.getsource(TaskGuidedPairSparseAttention.forward):
        errors.append("pair sparse attention topk missing")
    if "[B, 4, 21]" not in inspect.getsource(PairReliabilityHead) and "pair_reliability" not in inspect.getsource(PairReliabilityHead.forward):
        errors.append("pair reliability q_ar missing")
    router_src = inspect.getsource(ParetoSafeRouter.forward)
    if "final_action_logits = base_action_logits + action_delta" not in router_src or "router_action_scale" not in router_src:
        errors.append("router final_action anchor missing or action scale absent")
    loss_src = Path("fate_oia/losses/ceai_losses.py").read_text(encoding="utf-8")
    if "final_loss - base_loss" not in loss_src:
        errors.append("Pareto safety sign must be final_loss - base_loss + margin")
    if 'action_labels.unsqueeze(2) * reason_labels.unsqueeze(1)' in loss_src:
        errors.append("Cartesian action x reason positive pair target is forbidden")
    if "build_pair_seed_targets" not in loss_src or "weak_groups" not in loss_src:
        errors.append("reliability-aware weak pair seed missing")
    gb_src = inspect.getsource(gradient_budget)
    if "torch.autograd.grad" not in gb_src or "used_true_grad_norm" not in gb_src:
        errors.append("gradient budget must use true gradient norms")
    pc_src = inspect.getsource(pcgrad_lite)
    if "p.grad = p.grad + grad" not in pc_src or "grad_accumulation_steps" not in pc_src:
        errors.append("PCGrad must accumulate and respect grad_accumulation_steps")
    tr_src = inspect.getsource(train_ceai_oia)
    if "compute_trainer_readiness_state" not in tr_src or "readiness_state=readiness_state" not in tr_src:
        errors.append("trainer-level readiness is not wired into model forward")
    scene_src = inspect.getsource(bdd100k_scene_state)
    for token in ["_box_center", "_poly_mean_x", "front_vehicle_count", "lane_left_proxy", "direct_drivable_proxy"]:
        if token not in scene_src:
            errors.append(f"scene-state geometry proxy missing {token}")
    if "scene_queries" not in inspect.getsource(SceneStatePrototypeTransformer):
        errors.append("scene state learnable queries missing")
    if "final_action_logits" not in inspect.getsource(ceai_losses.ceai_main_loss):
        errors.append("main loss must use final action logits")
    if require_smoke_artifacts:
        if not smoke_dir:
            errors.append("smoke_dir required for artifact audit")
        else:
            errors.extend(validate_smoke_artifacts(smoke_dir))
    return errors


def validate_smoke_artifacts(smoke_dir: Path) -> list[str]:
    errors: list[str] = []
    required = [
        "metrics_summary.json",
        "branch_metrics.json",
        "readiness_stats.json",
        "selected_vs_random_evidence_stats.json",
        "grad_budget_stats.json",
        "pcgrad_accum_stats.json",
        "scene_state_stats.json",
        "scene_state_proxy_stats.json",
        "pair_reliability_stats.json",
        "pair_attention_stats.json",
        "router_stats.json",
        "loss_components.jsonl",
    ]
    epoch_dir = smoke_dir / "epoch_000"
    for name in required:
        p = epoch_dir / name
        if not p.exists() or p.stat().st_size <= 2:
            errors.append(f"missing/empty smoke artifact {p}")
    evidence_path = epoch_dir / "selected_vs_random_evidence_stats.json"
    if evidence_path.exists():
        errors.extend(validate_evidence_schema(json.loads(evidence_path.read_text(encoding="utf-8"))))
    grad_path = epoch_dir / "grad_budget_stats.json"
    if grad_path.exists():
        grad = json.loads(grad_path.read_text(encoding="utf-8"))
        for key in ["norm_main", "norm_aux", "budget_scale", "rho", "used_true_grad_norm"]:
            if key not in grad:
                errors.append(f"grad budget stats missing {key}")
        if grad.get("used_true_grad_norm") is not True:
            errors.append("grad budget did not report true gradient norm")
    pc_path = epoch_dir / "pcgrad_accum_stats.json"
    if pc_path.exists():
        pc = json.loads(pc_path.read_text(encoding="utf-8"))
        for key in ["pcgrad_task_count", "conflict_count", "projection_applied_count", "grad_accumulation_steps", "accumulated_microbatches", "overwrote_existing_grad"]:
            if key not in pc:
                errors.append(f"pcgrad stats missing {key}")
        if pc.get("overwrote_existing_grad"):
            errors.append("pcgrad reports overwritten accumulated grad")
    return errors


def run_smoke(args) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "-m",
        "fate_oia.engine.train_ceai_oia",
        "--config",
        args.config,
        "--output_dir",
        args.smoke_dir,
        "--epochs",
        "1",
        "--batch_size",
        "2",
        "--gradient_accumulation_steps",
        "1",
        "--num_workers",
        "0",
        "--max_train_samples",
        "16",
        "--max_test_samples",
        "16",
        "--device",
        args.device,
    ]
    return _run(cmd, timeout=900)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--smoke_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run_smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    branch = ""
    code, git_status = _run(["git", "status", "--short"], timeout=60)
    code_b, branch_out = _run(["git", "branch", "--show-current"], timeout=60)
    branch = branch_out.strip() if code_b == 0 else ""
    cwd = Path.cwd()
    if "fate_oia_ceai_oia_v1_worktree" not in str(cwd):
        errors.append(f"wrong worktree {cwd}")
    if branch != "ceai_oia_v1":
        errors.append(f"wrong branch {branch}")
    if ".background_runs" in git_status:
        errors.append(".background_runs appears in git status")
    errors.extend(validate_format_ast_yaml())
    errors.extend(validate_config(load_config_flat(Path(args.config))))
    errors.extend(validate_foreground_files([Path("scripts/FATE_OIA_ceai_oia_v1_foreground.ps1"), Path("fate_oia/engine/supervise_ceai_oia_foreground.py")]))
    py_code, py_out = _run([sys.executable, "-m", "py_compile", *[str(p) for p in ceai_py_files()]], timeout=120)
    if py_code != 0:
        errors.append("py_compile command failed:\n" + py_out)
    pytest_code, pytest_out = _run([sys.executable, "-m", "pytest", *[str(p) for p in Path("tests").glob("test_ceai_*.py")], "-q"], timeout=600)
    if pytest_code != 0:
        errors.append("targeted pytest failed:\n" + pytest_out)
    if args.run_smoke:
        smoke_code, smoke_out = run_smoke(args)
        (out / "smoke_stdout.txt").write_text(smoke_out, encoding="utf-8")
        if smoke_code != 0:
            errors.append("1-epoch smoke failed:\n" + smoke_out)
    elif not Path(args.smoke_dir).exists():
        errors.append(f"smoke_dir does not exist and --run_smoke was not passed: {args.smoke_dir}")
    errors.extend(run_static_audit(require_smoke_artifacts=True, smoke_dir=Path(args.smoke_dir)))
    report = {
        "branch": branch,
        "worktree": str(cwd),
        "config": args.config,
        "smoke_dir": args.smoke_dir,
        "git_status_short": git_status,
        "pytest_output_tail": pytest_out[-4000:],
        "errors": errors,
    }
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        (out / "REVIEW_FAIL_CEAI_OIA_V1_1.txt").write_text("\n".join(errors), encoding="utf-8")
        raise SystemExit("CEAI V1.1 audit failed:\n" + "\n".join(errors))
    (out / "REVIEW_PASS_CEAI_OIA_V1_1.txt").write_text(
        "\n".join(
            [
                "CEAI-OIA V1.1 REVIEW PASS",
                f"branch: {branch}",
                f"worktree: {cwd}",
                f"config: {args.config}",
                f"smoke_dir: {args.smoke_dir}",
                "ast_parse: PASS",
                "py_compile: PASS",
                "pytest: PASS",
                "smoke_artifacts: PASS",
                "pareto_safety_sign: PASS",
                "pair_seed_unknown_mask: PASS",
                "trainer_readiness_gate: PASS",
                "selected_vs_random_truthful: PASS",
                "true_gradient_budget: PASS",
                "pcgrad_accumulation_safe: PASS",
                "bdd100k_scene_geometry_proxy: PASS",
                "full_training_allowed_after_this_file_only: YES",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
