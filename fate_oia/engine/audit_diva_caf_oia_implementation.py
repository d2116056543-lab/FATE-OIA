from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SOURCE_TOKENS = {
    "fate_oia/datasets/diva_caf_oia_dataset.py": ["normalize_mean", "normalize_std", "patch_grid"],
    "fate_oia/models/diva_caf_oia_model.py": ["FATEOIAFeatureModel", "z_fate_action_logits", "action_fused_logits", "z_actor_action_logits", "selected_factor_meta", "no_test_leakage_assertion", "exp_prior", "exp_reliability", "factor_reason_support", "z_actor_without_selected"],
    "fate_oia/models/diva_visual_mixture_gate.py": ["z_fate + visual_gate * bounded_delta", "y_action", "delta_cap"],
    "fate_oia/engine/eval_diva_caf_oia.py": ["branch_safe_guarded_action", "guarded_source", "actor_minus_fate_mF1"],
    "fate_oia/models/diva_multilayer_dino.py": ["_forward_explicit_layers", "layer_stats", "extractor_type", "real_dino"],
    "fate_oia/models/diva_visual_actor.py": ["score_from_action_evidence", "z_eva_without_action_set", "z_eva_action_set_delta", "evidence_scale_usage"],
    "fate_oia/models/diva_deformable_attention.py": ["grid_sample"],
    "fate_oia/models/diva_action_set_transformer.py": ["register_buffer", "prototype_vectors", "TransformerEncoder"],
    "fate_oia/models/caf_factor_bank.py": ["factor_region", "factor_action_origin", "box"],
    "fate_oia/models/caf_bilevel_routing.py": ["sparsemax", "update_reliability", "lambda_exp", "hurt_ema", "exp_prior", "exp_reliability"],
    "fate_oia/models/caf_factor_auditor.py": ["action_gt_loss_drop_on_evidence_mask", "z_actor_without_selected", "z_actor_without_random"],
    "fate_oia/models/caf_reason_decoder.py": ["reason_to_factor_attention", "tail_reason_indices", "factor_reason_support"],
    "fate_oia/models/caf_reason_reliability.py": ["factor_support", "base_reason_logits is accepted", "not used as support"],
    "fate_oia/datasets/bdd100k_scene_state_proxy.py": ["box2d", "poly2d", "base_stem"],
    "fate_oia/losses/diva_caf_gradient_budget.py": ["torch.autograd.grad"],
    "fate_oia/losses/diva_caf_losses.py": ["asymmetric_loss_with_logits", "loss_type", "base_reason"],
    "fate_oia/engine/train_diva_caf_oia.py": ["resolve_config", "update_reliability", "history.json", "pretrained_weights", "test_ds", "no_feature_cache", "REVIEW_PASS_DIVA_CAF_OIA_V2_1"],
}

REQUIRED_ARTIFACTS = [
    "run_manifest.json", "metrics_latest.json", "branch_metrics_epoch_0.json", "visual_gate_stats.json",
    "diva_evidence_stats.json", "action_set_usage_stats.json", "factor_selection_stats.json",
    "factor_group_usage.json", "selected_vs_random_action_loss_drop.json", "reason_factor_attention_stats.json",
    "reason_tail_stats.json", "bdd100k_scene_state_stats.json", "no_test_leakage_assertion.json",
    "gradient_budget_stats.json", "visual_samples_epoch_0.jsonl", "history.json", "checkpoint_latest.pth", "checkpoint_best_test.pth",
    "dino_layer_stats.json", "exp_prior_stats.json", "lambda_exp_history.json", "guarded_action_stats.json",
]


def format_ast_yaml_audit(root: Path, config_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    files = list((root / "fate_oia").rglob("*diva*.py")) + list((root / "fate_oia").rglob("*caf*.py")) + list((root / "tests").glob("test_diva*.py")) + list((root / "tests").glob("test_caf*.py"))
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        if len(text.splitlines()) <= 5:
            failures.append(f"malformed one-line file: {path}")
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            failures.append(f"ast parse failed {path}: {exc}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        failures.append("config is not a YAML mapping")
    else:
        if cfg.get("training", {}).get("epochs") != 32:
            failures.append("config training.epochs must be 32")
        if cfg.get("data", {}).get("no_feature_cache") is not True:
            failures.append("config data.no_feature_cache must be true")
        if cfg.get("data", {}).get("eval_splits") != ["test"]:
            failures.append("config data.eval_splits must be ['test']")
        if not cfg.get("backbone", {}).get("pretrained_weights"):
            failures.append("config backbone.pretrained_weights must be nonempty for V2.1 real-DINO smoke")
    return {"passed": not failures, "failures": failures}


def static_source_audit(root: Path = Path(".")) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    failures: list[str] = []
    for rel, tokens in REQUIRED_SOURCE_TOKENS.items():
        path = root / rel
        if not path.exists():
            checks[rel] = {"exists": False}
            failures.append(f"missing {rel}")
            continue
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text)
            ast_ok = True
        except SyntaxError as exc:
            ast_ok = False
            failures.append(f"ast parse failed {rel}: {exc}")
        token_ok = {tok: (tok in text) for tok in tokens}
        for tok, ok in token_ok.items():
            if not ok:
                failures.append(f"missing token {tok} in {rel}")
        checks[rel] = {"exists": True, "ast_ok": ast_ok, "tokens": token_ok}
    sup = root / "fate_oia/engine/supervise_diva_caf_oia_foreground.py"
    if sup.exists():
        text = sup.read_text(encoding="utf-8")
        forbidden = ["Start" + "-Process", "Start" + "-Job", "no" + "hup", "WindowStyle" + " Hidden"]
        bad = [x for x in forbidden if x in text]
        if bad:
            failures.append(f"supervisor forbidden tokens: {bad}")
    else:
        failures.append("missing supervisor")
    model_text = (root / "fate_oia/models/diva_caf_oia_model.py").read_text(encoding="utf-8")
    if '"guarded_action_logits": z_actor' in model_text:
        failures.append("model still sets guarded_action_logits directly to z_actor")
    auditor_text = (root / "fate_oia/models/caf_factor_auditor.py").read_text(encoding="utf-8")
    if "delta *" in auditor_text or "masked_delta" in auditor_text:
        failures.append("factor auditor still uses delta scaling instead of evidence deletion")
    return {"checks": checks, "failures": failures, "passed": len(failures) == 0}


def artifact_audit(smoke_dir: Path) -> dict[str, Any]:
    failures = []
    details = {}
    for name in REQUIRED_ARTIFACTS:
        p = smoke_dir / name
        details[name] = {"exists": p.exists(), "size": p.stat().st_size if p.exists() else 0}
        if not p.exists() or p.stat().st_size <= 2:
            failures.append(f"missing/empty artifact {name}")
    p = smoke_dir / "selected_vs_random_action_loss_drop.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("method") != "action_gt_loss_drop_on_evidence_mask":
            failures.append("selected-vs-random artifact is not action-GT evidence-mask loss drop")
    p = smoke_dir / "no_test_leakage_assertion.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("used_bdd100k_gt_in_test_forward") is not False:
            failures.append("test forward leakage flag is not false")
    p = smoke_dir / "dino_layer_stats.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("extractor_type") != "real_dino":
            failures.append("review smoke did not use real_dino extractor")
        stats = data.get("layer_stats") or {}
        if not all(str(k) in stats or int(k) in stats for k in [3, 6, 9, 12]):
            failures.append("DINO layer stats missing one of [3,6,9,12]")
    return {"details": details, "failures": failures, "passed": len(failures) == 0}


def run_py_compile(root: Path) -> dict[str, Any]:
    files = [str(p) for p in (root / "fate_oia").rglob("*.py") if any(part in str(p) for part in ["diva", "caf"])]
    proc = subprocess.run([sys.executable, "-m", "py_compile", *files], cwd=str(root), text=True, capture_output=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "passed": proc.returncode == 0}


def run_targeted_pytest(root: Path) -> dict[str, Any]:
    files = [str(p) for p in (root / "tests").glob("test_diva_*.py")] + [str(p) for p in (root / "tests").glob("test_caf_*.py")]
    proc = subprocess.run([sys.executable, "-m", "pytest", *files, "-q"], cwd=str(root), text=True, capture_output=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "passed": proc.returncode == 0}


def run_smoke_if_needed(root: Path, smoke_dir: Path, device: str) -> dict[str, Any]:
    if (smoke_dir / "metrics_latest.json").exists():
        return {"skipped": True, "passed": True, "reason": "existing smoke artifacts"}
    cmd = [
        sys.executable, "-m", "fate_oia.engine.train_diva_caf_oia",
        "--config", "configs/fate_oia_train_360x640_diva_caf_oia_v2.yaml",
        "--output_dir", str(smoke_dir),
        "--epochs", "1",
        "--batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_train_samples", "8",
        "--max_test_samples", "8",
        "--device", device,
        "--no_feature_cache",
        "--test_only",
        "--print_every", "1",
    ]
    proc = subprocess.run(cmd, cwd=str(root), text=True, capture_output=True)
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "passed": proc.returncode == 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_diva_caf_oia_v2.yaml")
    parser.add_argument("--smoke_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    smoke_dir = Path(args.smoke_dir)
    config_path = root / args.config
    format_result = format_ast_yaml_audit(root, config_path)
    static = static_source_audit(root)
    compile_result = run_py_compile(root)
    pytest_result = run_targeted_pytest(root)
    smoke_result = run_smoke_if_needed(root, smoke_dir, args.device)
    artifacts = artifact_audit(smoke_dir)
    report = {"format_ast_yaml": format_result, "static": static, "py_compile": compile_result, "pytest": pytest_result, "smoke": smoke_result, "artifacts": artifacts}
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if format_result["passed"] and static["passed"] and compile_result["passed"] and pytest_result["passed"] and smoke_result["passed"] and artifacts["passed"]:
        (out / "REVIEW_PASS_DIVA_CAF_OIA_V2_1.txt").write_text("REVIEW_PASS_DIVA_CAF_OIA_V2_1\n", encoding="utf-8")
        print("REVIEW_PASS_DIVA_CAF_OIA_V2_1")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
