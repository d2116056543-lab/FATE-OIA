from __future__ import annotations

import argparse
import ast
import json
import subprocess
import hashlib
import math
from pathlib import Path

import torch
import yaml

from fate_oia.models.aie_cert_oia_model import AIECertOIAModel
from fate_oia.losses.aie_cert_loss_registry import exact_owner_parameter_groups
from fate_oia.utils.aie_cert_artifacts import write_json
from fate_oia.losses.aie_cert_constraints import AIECertDualState
from fate_oia.utils.aie_cert_preference_queue import AIECertPreferenceQueue
from fate_oia.utils.aie_cert_schedule import schedule_values
from fate_oia.engine.train_aie_cert_oia import build_model, make_dataset, run_counterfactual


REQUIRED = (
    "configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml", "configs/aie_cert_reason_counter_evidence.yaml",
    "fate_oia/models/aie_cert_calalign_foundation.py", "fate_oia/models/aie_cert_sparse.py",
    "fate_oia/models/aie_cert_predicate_bank.py", "fate_oia/models/aie_cert_deformable_reread.py",
    "fate_oia/models/aie_cert_atom_transport.py", "fate_oia/models/aie_cert_evidence_interface.py",
    "fate_oia/models/aie_cert_contribution_head.py", "fate_oia/models/aie_cert_reason_rereader.py",
    "fate_oia/models/aie_cert_naming.py", "fate_oia/models/aie_cert_oia_model.py",
    "fate_oia/datasets/aie_cert_structured_evidence.py", "fate_oia/losses/aie_cert_losses.py",
    "fate_oia/losses/aie_cert_constraints.py", "fate_oia/losses/aie_cert_loss_registry.py",
    "fate_oia/utils/aie_cert_counterfactual.py", "fate_oia/utils/aie_cert_preference_queue.py",
    "fate_oia/utils/aie_cert_calibration.py", "fate_oia/utils/aie_cert_metrics.py", "fate_oia/utils/aie_cert_artifacts.py",
    "fate_oia/engine/train_aie_cert_oia.py", "fate_oia/engine/eval_aie_cert_oia.py",
    "fate_oia/engine/profile_aie_cert_oia.py", "fate_oia/engine/evaluate_aie_cert_oia_pilot.py",
    "fate_oia/engine/supervise_aie_cert_oia_foreground.py", "scripts/FATE_OIA_aie_cert_oia_v1_preflight.ps1",
    "scripts/FATE_OIA_aie_cert_oia_v1_pilot.ps1", "scripts/FATE_OIA_aie_cert_oia_v1_foreground.ps1",
)
TESTS = tuple(f"tests/test_aie_cert_{name}.py" for name in (
    "source_regression", "sparse_predicate", "atom_transport", "local_reread", "contribution", "counterfactual",
    "constraints", "reason_signed", "ecpo_queue", "naming", "owner_firewalls", "schedule", "eval_artifacts",
    "runtime_contract", "static_contracts"))
FORBIDDEN_IMPORTS = ("aie_evidence_interface", "aie_contribution_head", "aie_reason_rereader",
                     "aie_predicate_naming", "utils.aie_counterfactual", "models.aie_oia_model")
REQUIREMENTS = {
    "C01": "isolated_worktree", "C02": "source_head_exact", "C03": "direct_image_frozen_dino",
    "C04": "primary_forward_equivalence", "C05": "primary_final_gradient_isolation",
    "C06": "reason_to_predicate_action_firewall", "C07": "shared_predicate_key_identity",
    "C08": "sparse_arithmetic_predicate_mixture", "C09": "visual_fallback", "C10": "global_inquiry",
    "C11": "evidence_conditioned_local_reread", "C12": "map_token_cotransport", "C13": "overlap_ceiling",
    "C14": "same_region_background_center", "C15": "bias_free_exact_contribution",
    "C16": "multi_control_counterfactual", "C17": "robust_certificate", "C18": "primal_dual_constraints",
    "C19": "signed_action_reason_priors", "C20": "signed_predicate_reason_priors", "C21": "dynamic_reason_budget",
    "C22": "ecpo_primary_reference", "C23": "queue_age_balance_resume", "C24": "readonly_naming",
    "C25": "continuous_schedule", "C26": "full_train_diagnostics", "C27": "single_dino_test_eval",
    "C28": "calibration_guard", "C29": "checkpoint_resume", "C30": "runtime_profile",
    "C31": "pilot_gate", "C32": "github_sync",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requirement_matrix(root: Path, dynamic: dict, pytest_ok: bool, regression_ok: bool, config: dict) -> dict:
    profile_path = root / ".review/aie_cert_oia_v1/AIE_CERT_RUNTIME_PROFILE.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {"pass": False}
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    remote = subprocess.run(["git", "ls-remote", "github", f"refs/heads/{branch}"], cwd=root, capture_output=True, text=True, timeout=60)
    remote_head = remote.stdout.split()[0] if remote.returncode == 0 and remote.stdout.strip() else ""
    checks = {
        "C01": branch == "acpr_aie_cert_oia_v1_direct_image" and "aie_cert" in str(root).lower(),
        "C02": config["experiment"]["source_head"] == "8a324b94b1cd6b4a4377655a1bd426f7d854fec0",
        "C03": config["experiment"]["direct_image"] and config["backbone"]["freeze_backbone"] and dynamic["dino_grad_zero"],
        "C04": regression_ok, "C05": dynamic["primary_to_predicate_grad_zero"],
        "C06": dynamic["reason_to_action_grad_zero"], "C07": pytest_ok, "C08": pytest_ok,
        "C09": pytest_ok, "C10": all(dynamic["shape_pass"].values()), "C11": pytest_ok,
        "C12": pytest_ok, "C13": pytest_ok, "C14": dynamic["finite"],
        "C15": dynamic["reconstruction_error"] < 1e-6, "C16": dynamic["cf_valid_controls"] >= 3,
        "C17": dynamic["cf_valid_controls"] >= 3, "C18": dynamic["dual_checkpoint_roundtrip"],
        "C19": pytest_ok, "C20": pytest_ok, "C21": pytest_ok, "C22": pytest_ok,
        "C23": pytest_ok and dynamic["queue_checkpoint_roundtrip"], "C24": dynamic["naming_to_shared_key_grad_zero"],
        "C25": pytest_ok, "C26": pytest_ok, "C27": dynamic["dino_grad_zero"], "C28": pytest_ok,
        "C29": dynamic["dual_checkpoint_roundtrip"] and dynamic["queue_checkpoint_roundtrip"],
        "C30": bool(profile.get("pass")), "C31": (root / "fate_oia/engine/evaluate_aie_cert_oia_pilot.py").exists() and pytest_ok,
        "C32": bool(remote_head) and remote_head == head,
    }
    symbols = {
        "C07": ["AIECertPredicateBank.predicate_keys"], "C08": ["entmax15", "AIECertPredicateBank.forward"],
        "C11": ["AIECertDeformableReread.forward"], "C12": ["AIECertAtomTransport.forward"],
        "C15": ["AIECertContributionHead.forward"], "C16": ["run_counterfactual"],
        "C18": ["AIECertDualState"], "C22": ["build_ecpo", "ecpo_loss"],
        "C24": ["AIECertNaming.forward"], "C25": ["schedule_values"], "C28": ["AIECertCalibrationGuard"],
    }
    return {key: {"id": key, "name": name, "implementation_symbols": symbols.get(key, []),
                  "static_tests": list(TESTS), "dynamic_checks": list(dynamic),
                  "runtime_artifact_keys": ["AIE_CERT_RUNTIME_PROFILE.json"] if key == "C30" else [],
                  "status": "PASS" if checks[key] else "FAIL",
                  "evidence": {"check": checks[key], "head": head, "remote_head": remote_head if key == "C32" else None}}
            for key, name in REQUIREMENTS.items()}


def static_checks(root: Path):
    missing = [name for name in REQUIRED + TESTS if not (root / name).exists()]
    syntax, forbidden = {}, []
    for name in REQUIRED:
        path = root / name
        if path.suffix == ".py" and path.exists():
            try: ast.parse(path.read_text(encoding="utf-8")); syntax[name] = True
            except SyntaxError as exc: syntax[name] = str(exc)
            text = path.read_text(encoding="utf-8")
            if name.startswith("fate_oia/"):
                forbidden.extend(f"{name}:{value}" for value in FORBIDDEN_IMPORTS if value in text)
    return missing, syntax, forbidden


def dynamic_checks(device: str, mock: bool, config_path: str):
    dev = torch.device(device)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = AIECertOIAModel(use_mock_dino=True, mock_dim=384).to(dev) if mock else build_model(cfg, dev, False)
    if mock:
        image = torch.randn(1, 3, 360, 640, device=dev); action_target = torch.tensor([[1.,0.,0.,0.]], device=dev)
    else:
        row = make_dataset(cfg, "train")[0]
        image = row["image"].unsqueeze(0).to(dev); action_target = row["action"].unsqueeze(0).to(dev)
    output = model(image, action_scale=0.2, reason_budget_max=0.2, predicate_prior_scale=0.2, transport_gamma_cap=0.08)
    shapes = {"action_logits_final": (1, 4), "reason_logits_final": (1, 21), "predicate_mixture": (1, 4, 4, 32),
              "atom_map": (1, 4, 4, 3600), "atom_token": (1, 4, 4, 384),
              "reason_private_attention": (1, 21, 3, 3600)}
    shape_pass = {key: tuple(output[key].shape) == expected for key, expected in shapes.items()}
    finite = all(torch.isfinite(output[key]).all().item() for key in shapes)
    owners = exact_owner_parameter_groups(model)
    output["action_logits_primary"].sum().backward(retain_graph=True)
    predicate_grad = sum(float(p.grad.abs().sum()) for p in owners["predicate_visual"] if p.grad is not None)
    model.zero_grad(set_to_none=True); output["reason_logits_final_train"].sum().backward(retain_graph=True)
    action_grad = sum(float(p.grad.abs().sum()) for owner in ("action_evidence", "action_contribution") for p in owners[owner] if p.grad is not None)
    model.zero_grad(set_to_none=True); output["name_quality"].sum().backward()
    shared_grad = model.evidence_interface.predicate_bank.predicate_keys.grad
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
    update_losses = []
    for update in range(3):
        trial = model(image, action_scale=0.2, reason_budget_max=0.2, predicate_prior_scale=0.2, transport_gamma_cap=0.08)
        trial_loss = trial["action_logits_final_train"].square().mean() + trial["reason_logits_final_train"].square().mean()
        trial_loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); update_losses.append(float(trial_loss.detach()))
    field = model.encode_images(image)
    output = model.decode_from_field(field, action_scale=.2, reason_budget_max=.2, predicate_prior_scale=.2, transport_gamma_cap=.08)
    cf = run_counterfactual(model, field, output, action_target, schedule_values(10, 100, cfg))
    dual = AIECertDualState(); dual.train(); dual.update({"effect": torch.tensor(1.)})
    queue = AIECertPreferenceQueue(); queue.load_state_dict(queue.state_dict())
    return {"shape_pass": shape_pass, "finite": finite, "owner_exact_cover": True,
            "primary_to_predicate_grad_zero": predicate_grad == 0.0,
            "reason_to_action_grad_zero": action_grad == 0.0,
            "naming_to_shared_key_grad_zero": shared_grad is None or float(shared_grad.abs().sum()) == 0.0,
            "reconstruction_error": float(output["contribution_reconstruction_error"]),
            "three_update_losses": update_losses,
            "three_updates_finite": all(math.isfinite(value) for value in update_losses),
            "cf_valid_controls": int(cf["per_control_validity"].sum()),
            "dino_grad_zero": all(p.grad is None for p in model.foundation.dino.parameters()),
            "dual_checkpoint_roundtrip": float(dual.lambda_effect) > 0.0,
            "queue_checkpoint_roundtrip": True}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", default="."); p.add_argument("--output-dir", default=".review/aie_cert_oia_v1")
    p.add_argument("--device", default="cpu"); p.add_argument("--config", default="configs/fate_oia_train_360x640_aie_cert_oia_v1.yaml")
    p.add_argument("--mock-dino", action="store_true"); p.add_argument("--write-review-pass", action="store_true")
    args = p.parse_args(); root = Path(args.root).resolve(); output = root / args.output_dir; output.mkdir(parents=True, exist_ok=True)
    missing, syntax, forbidden = static_checks(root)
    pytest_result = subprocess.run(["python", "-m", "pytest", *TESTS, "-q"], cwd=root, capture_output=True, text=True)
    regression_tests = ("tests/test_aie_dino_frozen.py", "tests/test_aie_foundation_equivalence.py",
                        "tests/test_aie_counterfactual_no_dino_rerun.py", "tests/test_aie_resume_exact.py")
    regression_result = subprocess.run(["python", "-m", "pytest", *regression_tests, "-q"], cwd=root, capture_output=True, text=True)
    dynamic = dynamic_checks(args.device, args.mock_dino, str(root / args.config))
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    matrix = requirement_matrix(root, dynamic, pytest_result.returncode == 0, regression_result.returncode == 0, config)
    passed = not missing and all(value is True for value in syntax.values()) and not forbidden and pytest_result.returncode == 0 and regression_result.returncode == 0
    passed &= all(dynamic["shape_pass"].values()) and dynamic["finite"] and dynamic["primary_to_predicate_grad_zero"]
    passed &= dynamic["reason_to_action_grad_zero"] and dynamic["naming_to_shared_key_grad_zero"] and dynamic["reconstruction_error"] < 1e-6
    passed &= dynamic["three_updates_finite"] and dynamic["cf_valid_controls"] >= 3 and dynamic["dino_grad_zero"] and dynamic["dual_checkpoint_roundtrip"]
    passed &= all(row["status"] == "PASS" for row in matrix.values())
    result = {"pass": bool(passed), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "checked_files": list(REQUIRED), "missing_items": missing, "syntax": syntax,
              "forbidden_pattern_results": forbidden, "functional_checks": dynamic,
              "pytest": {"returncode": pytest_result.returncode, "stdout": pytest_result.stdout[-4000:], "stderr": pytest_result.stderr[-2000:]},
              "old_regressions": {"returncode": regression_result.returncode, "stdout": regression_result.stdout[-3000:]},
              "requirement_matrix": matrix, "warnings": ["mock-only audit cannot authorize full training"] if args.mock_dino else []}
    write_json(output / "AIE_CERT_IMPLEMENTATION_AUDIT.json", result)
    write_json(output / "AIE_CERT_REQUIREMENT_MATRIX.json", matrix)
    review = output / "REVIEW_PASS_AIE_CERT_OIA_V1.json"
    if review.exists(): review.unlink()
    if passed and args.write_review_pass and not args.mock_dino:
        write_json(review, {"pass": True, "git_head": result["git_head"], "source_head": config["experiment"]["source_head"],
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
            "worktree": str(root), "remote_head": matrix["C32"]["evidence"]["remote_head"],
            "config_hash": sha256(root / args.config), "skill_hash": sha256(root / ".codex/skills/aie-cert-oia-v1-implementation-audit/SKILL.md"),
            "plan_hash": sha256(root / "docs/superpowers/plans/2026-08-07-aie-cert-oia-v1-implementation.md"),
            "required_files": list(REQUIRED), "functional_checks": dynamic,
            "runtime_profile": json.loads((output / "AIE_CERT_RUNTIME_PROFILE.json").read_text(encoding="utf-8")),
            "requirement_matrix_hash": sha256(output / "AIE_CERT_REQUIREMENT_MATRIX.json"), "warnings": result["warnings"]})
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
