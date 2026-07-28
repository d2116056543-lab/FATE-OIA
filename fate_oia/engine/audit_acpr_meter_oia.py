from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.losses.meter_pu_losses import meter_private_pu_loss
from fate_oia.utils.meter_artifacts import (
    combined_file_hash,
    file_hash,
    python_source_tree_hash,
    state_hash,
    write_json,
)
from fate_oia.utils.meter_config import load_meter_config


REQUIRED_FILES = [
    "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml",
    "configs/meter_factor_schema.yaml",
    "configs/meter_grounding_schema.yaml",
    "fate_oia/datasets/meter_grounding_index.py",
    "fate_oia/datasets/meter_signed_targets.py",
    "fate_oia/datasets/meter_dataset.py",
    "fate_oia/transforms_meter.py",
    "fate_oia/models/meter_calalign_foundation.py",
    "fate_oia/models/meter_signed_factors.py",
    "fate_oia/models/meter_semantic_action.py",
    "fate_oia/models/meter_reason_decoder.py",
    "fate_oia/models/meter_meta_adapters.py",
    "fate_oia/models/meter_oia_model.py",
    "fate_oia/losses/meter_action_losses.py",
    "fate_oia/losses/meter_reason_losses.py",
    "fate_oia/losses/meter_grounding_losses.py",
    "fate_oia/losses/meter_counterfactual_losses.py",
    "fate_oia/losses/meter_pu_losses.py",
    "fate_oia/optim/meter_meta_utility.py",
    "fate_oia/utils/meter_posthoc_calibration.py",
    "fate_oia/utils/meter_artifacts.py",
    "fate_oia/utils/meter_runtime.py",
    "fate_oia/engine/train_acpr_meter_oia.py",
    "fate_oia/engine/eval_acpr_meter_oia.py",
    "fate_oia/engine/audit_acpr_meter_oia.py",
    "fate_oia/engine/profile_acpr_meter_oia.py",
    "fate_oia/engine/supervise_acpr_meter_oia_foreground.py",
    "fate_oia/engine/export_meter_cases.py",
    "scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1",
    ".codex/skills/meter-oia-v1-implementation-audit/SKILL.md",
]
FORBIDDEN = (
    "ACPRPairMemory", "matched_pair", "action_set_probs @", "cached_logits", "feature_cache_enabled: true",
    "token_compression: keep_merge", "checkpoint_best_val", "eval_splits: val", "best_selection_split: val",
    "Start-Process", "Start-Job", "scheduled task", "hidden process", "daemon",
)


def _python_files(root: Path) -> list[Path]:
    return [root / path for path in REQUIRED_FILES if path.endswith(".py") and (root / path).exists()]


def _ast_placeholders(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            errors.append(f"{path}:{node.lineno}: pass")
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
            errors.append(f"{path}:{node.lineno}: NotImplementedError")
    return errors


def _dynamic_checks(
    device: torch.device,
    *,
    use_mock_dino: bool,
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    batch_size = 2 if use_mock_dino else 1
    model = METEROIAModel(
        use_mock_dino=use_mock_dino,
        pretrained_weights=(
            config["backbone"]["pretrained_weights"]
            if config is not None
            else "ckp/reference/dino_deitsmall8_pretrain.pth"
        ),
    ).to(device)
    images = torch.randn(batch_size, 3, 360, 640, device=device)
    model.eval()
    with torch.no_grad():
        field = model.encode_images(images)
        foundation = model.foundation.decode_foundation(field)
        out0 = model.decode_from_field(field, progress=0.0)
        out1_probe = model.decode_from_field(field, progress=1.0)
        selector_visual = model.decode_from_field(
            field,
            progress=1.0,
            diagnostic_modes=("selector_visual_only",),
        )
        selector_semantic = model.decode_from_field(
            field,
            progress=1.0,
            diagnostic_modes=("selector_semantic_only",),
        )
    out1 = model.decode_from_field(field, progress=1.0)
    factor_off = model.decode_from_field(field, progress=1.0, diagnostic_modes=("factor_off",))
    meta_off = model.decode_from_field(field, progress=1.0, diagnostic_modes=("meta_off",))
    required_shapes = {
        "patch_tokens_by_layer": [batch_size, 3, 3600, 384],
        "label_nodes": [batch_size, 25, 384],
        "action_logits_final": [batch_size, 4],
        "reason_logits_final": [batch_size, 21],
        "factor_support_map": [batch_size, 21, 3600],
        "action_factor_contributions": [batch_size, 4, 21],
    }
    shape_result = {name: list(out0.get(name, foundation.get(name)).shape) for name in required_shapes}
    shape_ok = all(shape_result[name] == expected for name, expected in required_shapes.items())
    layer_maps = out1_probe["factor_support_maps_by_layer"]
    null = out1_probe["factor_support_null_by_layer"]
    normalization = (layer_maps.sum(-1) + null).mean().item()
    support_counter_gap = (out1_probe["factor_support_map"] - out1_probe["factor_counter_map"]).abs().mean().item()
    contribution_error = (out1["action_logits_semantic"] - (model.action_peer.semantic_bias.view(1, -1) + out1["action_factor_contributions"].sum(-1))).abs().max().item()
    foundation_error = (out0["action_logits_final"] - foundation["action_logits_calalign"]).abs().max().item()
    reason_error = (out0["reason_logits_final"] - foundation["reason_logits_calalign"]).abs().max().item()
    loss = out1["action_logits_final"].square().mean() + out1["reason_logits_final"].square().mean()
    loss.backward()
    pu_logits = torch.zeros(2, 3, device=device, requires_grad=True)
    pu_targets = torch.ones(2, 3, device=device)
    pu_score = torch.ones(2, 3, device=device)
    pu_zero_loss = meter_private_pu_loss(pu_logits, pu_targets, pu_score, torch.zeros(3, device=device))
    pu_zero_loss.backward()
    pu_zero_lambda_disabled = float(pu_zero_loss.detach().cpu()) == 0.0 and bool(torch.equal(pu_logits.grad, torch.zeros_like(pu_logits)))
    dino_grad = max((float(p.grad.abs().max()) for p in model.foundation.dino.parameters() if p.grad is not None), default=0.0)
    downstream_grad = max((float(p.grad.abs().max()) for p in model.action_peer.parameters() if p.grad is not None), default=0.0)
    model.zero_grad(set_to_none=True)
    pu_output = model.decode_from_field(field, progress=1.0)
    pu_output["reason_logits_pu_private"].square().mean().backward()
    pu_meta_grad = max(
        (
            float(parameter.grad.abs().max())
            for parameter in model.signed_factors.meta_adapters.parameters()
            if parameter.grad is not None
        ),
        default=0.0,
    )
    pu_private_grad = max(
        (
            float(parameter.grad.abs().max())
            for parameter in model.reason_decoder.parameters()
            if parameter.grad is not None
        ),
        default=0.0,
    )
    return {
        "shape_result": shape_result,
        "shape_ok": shape_ok,
        "map_normalization_mean": normalization,
        "map_normalization_ok": abs(normalization - 1.0) < 1e-4,
        "support_counter_gap": support_counter_gap,
        "support_counter_separated": support_counter_gap > 1e-6,
        "semantic_additivity_error": contribution_error,
        "semantic_additivity_ok": contribution_error < 1e-6,
        "progress_zero_action_error": foundation_error,
        "progress_zero_reason_error": reason_error,
        "progress_zero_ok": foundation_error < 1e-6 and reason_error < 1e-6,
        "dino_grad_max": dino_grad,
        "downstream_grad_max": downstream_grad,
        "dino_frozen_ok": dino_grad == 0.0,
        "downstream_grad_ok": downstream_grad > 0.0,
        "pu_zero_lambda_disabled": pu_zero_lambda_disabled,
        "pu_private_firewall_ok": pu_meta_grad == 0.0 and pu_private_grad > 0.0,
        "selector_visual_only_error": float(
            (selector_visual["action_logits_final"] - selector_visual["action_logits_visual"]).abs().max()
        ),
        "selector_semantic_only_error": float(
            (selector_semantic["action_logits_final"] - selector_semantic["action_logits_semantic"]).abs().max()
        ),
        "ordinary_dino_calls": model.foundation.ordinary_dino_calls,
        "factor_off_action_delta": float((out1["action_logits_final"] - factor_off["action_logits_final"]).abs().mean().item()),
        "meta_off_reason_delta": float((out1["reason_logits_final"] - meta_off["reason_logits_final"]).abs().mean().item()),
        "all_required_outputs_finite": all(bool(torch.isfinite(value).all()) for value in (out1["action_logits_final"], out1["reason_logits_final"], out1["factor_support_map"], out1["action_factor_contributions"])),
        "real_dino_dynamic": not use_mock_dino,
    }


def run_audit(
    root: Path,
    output_dir: Path,
    *,
    device: str,
    real_dino: bool,
    profile_dir: Path,
    write_review_pass: bool,
    write_pre_pilot_ready: bool = False,
    test_summary_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checked = [path for path in REQUIRED_FILES if (root / path).exists()]
    missing = [path for path in REQUIRED_FILES if not (root / path).exists()]
    forbidden_hits: dict[str, list[str]] = {}
    for path in _python_files(root):
        if path.name == "audit_acpr_meter_oia.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [pattern for pattern in FORBIDDEN if pattern in text]
        if hits:
            forbidden_hits[str(path.relative_to(root))] = hits
    placeholder_errors: list[str] = []
    for path in _python_files(root):
        if path.name == "audit_acpr_meter_oia.py":
            continue
        placeholder_errors.extend(_ast_placeholders(path))
    config_error = None
    try:
        config = load_meter_config(root / "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml")
    except Exception as exc:
        config = None
        config_error = repr(exc)
    dynamic = _dynamic_checks(
        torch.device(device),
        use_mock_dino=not real_dino,
        config=config,
    )
    trainer_text = (root / "fate_oia/engine/train_acpr_meter_oia.py").read_text(encoding="utf-8") if (root / "fate_oia/engine/train_acpr_meter_oia.py").exists() else ""
    eval_text = (root / "fate_oia/engine/eval_acpr_meter_oia.py").read_text(encoding="utf-8") if (root / "fate_oia/engine/eval_acpr_meter_oia.py").exists() else ""
    counterfactual_text = (
        trainer_text.split("def _counterfactual_event(", 1)[1].split("def _make_optimizer(", 1)[0]
        if "def _counterfactual_event(" in trainer_text and "def _make_optimizer(" in trainer_text
        else ""
    )
    meta_text = (
        root / "fate_oia/optim/meter_meta_utility.py"
    ).read_text(encoding="utf-8")
    contract = {
        "trainer_calls_all_losses": all(token in trainer_text for token in ("meter_action_loss", "meter_reason_loss", "meter_grounding_loss", "meter_counterfactual_loss", "meter_private_pu_loss")),
        "trainer_calls_meta_and_calibration": all(token in trainer_text for token in ("meta.event", "_fit_calibration", "save_epoch_artifacts", "load_checkpoint")),
        "calibration_guard_called": all(token in trainer_text for token in ("guard_train_calib_deploy_theta", "fallback_on_deploy_degradation", "train_calib")),
        "pu_data_driven_not_fixed_training_zero": "pu_lambda = torch.zeros(reason_target.shape[1]" not in trainer_text and "meter_hidden_positive_audit" in trainer_text,
        "pu_zero_lambda_disabled": dynamic["pu_zero_lambda_disabled"],
        "counterfactual_same_field": (
            all(
                token in counterfactual_text
                for token in (
                    "delete_field",
                    "_select_mass_mask",
                    "_matched_control_mask",
                    "_replace_selected_with_neighbor_mean",
                    "wrong_output",
                )
            )
            and "encode_images" not in counterfactual_text
        ),
        "diagnostics_from_forward_tensors": all(token in eval_text for token in ("_mechanism_stats", "factor_support_map", "action_factor_contributions", "reason_mix_gate")),
        "resolved_yaml_written": all(token in trainer_text for token in ("config_resolved.yaml", "owner_manifest.json", "runtime_profile.json")),
        "resume_state_restored": all(
            token in trainer_text
            for token in (
                "--resume",
                "meta_state",
                "pu_state",
                "load_checkpoint",
                "resume_micro_step",
                "epoch_resume_micro",
                "micro_step=micro + 1",
                "torch.Generator().manual_seed",
            )
        ),
        "factor_off_effect_observable": dynamic["factor_off_action_delta"] > 0.0,
        "all_dynamic_outputs_finite": dynamic["all_required_outputs_finite"],
        "grounding_direction_and_mirror": all(
            token in (root / "fate_oia/losses/meter_grounding_losses.py").read_text(encoding="utf-8")
            for token in ("signed_direction", "_mirror_balance_loss", "factor_source_conf")
        ),
        "per_factor_meta_utility": all(
            token in meta_text
            for token in ("factor_ids", "utility = torch.stack(utilities)", "for factor_id in ids", "action_gradient[factor_id]")
        ),
        "matched_null_meta_admission": bool(
            config is not None
            and config.get("meta", {}).get("admission_mode")
            == "matched_null_lcb"
            and int(config.get("meta", {}).get("admission_audit_batches", 0))
            >= 2
            and int(config.get("meta", {}).get("admission_min_consecutive", 0))
            >= 2
            and all(
                token in meta_text
                for token in (
                    "_match_delta_norm",
                    "null_utility_samples",
                    "admission_lcb",
                    "null_q99",
                    "positive_streak",
                    "update_norm_normalized_utility",
                )
            )
            and all(
                token in trainer_text
                for token in (
                    "meta_audit_fields",
                    "audit_field_cache_reused",
                    "admission_score_ema",
                    "positive_streak",
                )
            )
        ),
        "private_reason_complete": all(
            token in (root / "fate_oia/models/meter_reason_decoder.py").read_text(encoding="utf-8")
            for token in ("reason_self_attention", "layer_router", "reason_logits_candidate")
        ),
        "observability_reason_weighting": "observability * (1.0 - evidence)" in (
            root / "fate_oia/losses/meter_reason_losses.py"
        ).read_text(encoding="utf-8"),
        "hidden_positive_subset_audit": all(
            token in (root / "fate_oia/losses/meter_pu_losses.py").read_text(encoding="utf-8")
            for token in ("deliberately_hidden_positive_vs_observed_zero", "audit_mask", "hidden_positive_auprc")
        ),
        "posthoc_search_complete": all(
            token in (root / "fate_oia/utils/meter_posthoc_calibration.py").read_text(encoding="utf-8")
            for token in ("label_groups", "group_shrinkage", "temperature", "threshold_rms_ratio", "map_max_abs_delta")
        ),
        "optimizer_contract": all(
            token in trainer_text
            for token in ("reference_effective_batch", "learning_rate_scale", "weight_decay", "allow_tf32")
        ),
        "profiler_contract": all(
            token in (root / "fate_oia/engine/profile_acpr_meter_oia.py").read_text(encoding="utf-8")
            for token in (
                "warmup_updates: int = 5",
                "measured_updates: int = 20",
                "optimizer.step()",
                "meta.event(",
                "_counterfactual_event(",
                '"real_data": True',
                '"--single_trial"',
                "subprocess.run(",
                "_wait_for_gpu_baseline(",
                '"isolation_pass"',
                '"memory_peak_after_ordinary_gb"',
                '"memory_peak_after_counterfactual_gb"',
                '"memory_peak_after_meta_gb"',
                '"memory_peak_after_calibration_gb"',
                '"git_head"',
                '"source_tree_hash"',
                '"config_hash"',
                '"schema_hash"',
            )
        ),
        "standalone_evaluator": all(
            token in eval_text
            for token in ("load_checkpoint(args.checkpoint", "evaluation_summary.json", "METERDataset(")
        ),
        "strict_full_readiness": all(
            token in trainer_text
            for token in ("METER_OIA_V1_FULL_TRAIN_READY.json", "_write_full_train_ready_if_eligible", "4096", "1024", "512")
        ),
    }
    real_result = {"required": True, "executed": False, "pass": False, "reason": "real-DINO profile not found"}
    profile_path = profile_dir / "runtime_profile.json"
    if real_dino and profile_path.exists():
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        selected = profile.get("selected") or {}
        recovery_samples = selected.get("physical_recovery_samples_gb") or []
        baseline_before = float(
            selected.get("physical_baseline_before_gb", -1.0)
        )
        recovery_verified = (
            len(recovery_samples) >= 3
            and baseline_before >= 0.0
            and all(
                float(sample) <= baseline_before + 0.5
                for sample in recovery_samples[-3:]
            )
        )
        profile_identity_ok = (
            int(profile.get("schema_version", 0)) == 1
            and profile.get("profile_mode")
            in {"candidate_search", "selected_validation"}
            and profile.get("git_head") == git_head
            and profile.get("source_tree_hash") == python_source_tree_hash(root)
            and profile.get("config_hash")
            == file_hash(root / "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml")
            and profile.get("schema_hash")
            == combined_file_hash(
                root / "configs/meter_factor_schema.yaml",
                root / "configs/meter_grounding_schema.yaml",
            )
        )
        real_result = {
            "required": True, "executed": bool(profile.get("real_dino")),
            "pass": (
                bool(profile.get("real_dino"))
                and bool(selected.get("finite"))
                and bool(selected.get("meta_utility_finite"))
                and bool(selected.get("isolation_pass"))
                and int(selected.get("child_exit_status", 1)) == 0
                and int(selected.get("process_id", 0)) > 0
                and recovery_verified
                and float(selected.get("physical_peak_delta_gb", -1.0)) >= 0.0
                and profile_identity_ok
                and float(selected.get("reserved_gb", 1e9)) < 45.0
                and all(
                    field in selected
                    for field in (
                        "memory_peak_after_ordinary_gb",
                        "memory_peak_after_counterfactual_gb",
                        "memory_peak_after_meta_gb",
                        "memory_peak_after_calibration_gb",
                    )
                )
            ),
            "profile_path": str(profile_path),
            "selected": selected,
            "identity_ok": profile_identity_ok,
            "recovery_verified": recovery_verified,
        }
    functional = {
        "dataset_targets": (root / "fate_oia/datasets/meter_signed_targets.py").exists(),
        "dino_field": dynamic["shape_ok"] and dynamic["dino_frozen_ok"],
        "signed_factor_maps": dynamic["map_normalization_ok"] and dynamic["support_counter_separated"],
        "semantic_action": dynamic["semantic_additivity_ok"],
        "foundation_equivalence": dynamic["progress_zero_ok"],
        "meta_override_reachable": "factor_parameter_override" in (root / "fate_oia/models/meter_oia_model.py").read_text(encoding="utf-8"),
        "formal_trainer": (root / "fate_oia/engine/train_acpr_meter_oia.py").exists(),
        "formal_evaluator": (root / "fate_oia/engine/eval_acpr_meter_oia.py").exists(),
        "foreground_supervisor": (root / "scripts/FATE_OIA_acpr_meter_oia_v1_foreground.ps1").exists(),
        "formal_loss_and_artifact_call_graph": contract["trainer_calls_all_losses"] and contract["trainer_calls_meta_and_calibration"] and contract["calibration_guard_called"] and contract["diagnostics_from_forward_tensors"],
        "runtime_and_resume_contract": contract["resolved_yaml_written"] and contract["resume_state_restored"],
        "counterfactual_contract": contract["counterfactual_same_field"],
        "pu_contract": contract["pu_data_driven_not_fixed_training_zero"] and contract["pu_zero_lambda_disabled"],
        "dynamic_finite_and_ablation": contract["all_dynamic_outputs_finite"] and contract["factor_off_effect_observable"],
        "grounding_objective": contract["grounding_direction_and_mirror"],
        "per_factor_meta_utility": contract["per_factor_meta_utility"],
        "matched_null_meta_admission": contract[
            "matched_null_meta_admission"
        ],
        "private_reason_decoder": contract["private_reason_complete"],
        "selective_observation_reason_loss": contract["observability_reason_weighting"],
        "hidden_positive_pu_audit": contract["hidden_positive_subset_audit"] and dynamic["pu_private_firewall_ok"],
        "posthoc_calibration": contract["posthoc_search_complete"],
        "optimizer_protocol": contract["optimizer_contract"],
        "runtime_profiler": contract["profiler_contract"],
        "standalone_evaluator": contract["standalone_evaluator"],
        "strict_full_readiness": contract["strict_full_readiness"],
        "selector_ablation": dynamic["selector_visual_only_error"] < 1e-7 and dynamic["selector_semantic_only_error"] < 1e-7,
    }
    clean_status = subprocess.run(["git", "status", "--porcelain"], cwd=root, text=True, capture_output=True, check=False).stdout.strip()
    passed = bool(
        not missing
        and not forbidden_hits
        and not placeholder_errors
        and config is not None
        and all(functional.values())
        and dynamic["downstream_grad_ok"]
        and real_result["pass"]
        and not clean_status
    )
    report = {
        "pass": passed,
        "checked_files": checked,
        "missing_items": missing + placeholder_errors,
        "forbidden_pattern_results": forbidden_hits,
        "functional_checks": functional,
        "dynamic_checks": dynamic,
        "contract_checks": contract,
        "real_dino": real_result,
        "config_error": config_error,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_tree_hash": python_source_tree_hash(root),
        "config_hash": file_hash(root / "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml"),
        "schema_hash": combined_file_hash(root / "configs/meter_factor_schema.yaml", root / "configs/meter_grounding_schema.yaml"),
        "warnings": ["No readiness artifact may be issued without real-DINO execution."] if not real_dino else [],
        "clean_head": not bool(clean_status),
        "clean_status": clean_status,
    }
    write_json(output_dir / "implementation_audit_METER_OIA_V1.json", report)
    if passed and write_review_pass:
        (output_dir / "REVIEW_PASS_METER_OIA_V1.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if passed and write_pre_pilot_ready:
        if clean_status:
            report["pass"] = False
            report["warnings"].append("Pre-pilot readiness requires a clean source/config/schema/skill HEAD.")
        elif test_summary_path is None or not test_summary_path.exists():
            report["pass"] = False
            report["warnings"].append("Pre-pilot readiness requires a recorded test summary.")
        else:
            test_summary = json.loads(test_summary_path.read_text(encoding="utf-8-sig"))
            ready = {
                "artifact": "METER_OIA_V1_PRE_PILOT_READY",
                "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
                "HEAD": report["git_head"],
                "base_sha": report["config_error"] is None and load_meter_config(root / "configs/fate_oia_train_360x640_acpr_meter_oia_v1.yaml").get("audit", {}).get("require_source_sha"),
                "source_tree_hash": report["source_tree_hash"],
                "config_hash": report["config_hash"],
                "schema_hash": report["schema_hash"],
                "skill_hash": file_hash(root / ".codex/skills/meter-oia-v1-implementation-audit/SKILL.md"),
                "audit_report": str(output_dir / "implementation_audit_METER_OIA_V1.json"),
                "real_dino": real_result,
                "dynamic_checks": dynamic,
                "test_summary": test_summary,
                "internal_test_selected": True,
                "publication_eligible": False,
                "unresolved": [],
            }
            write_json(root / ".review/METER_OIA_V1_PRE_PILOT_READY.json", ready)
            report["pre_pilot_ready_path"] = str(root / ".review/METER_OIA_V1_PRE_PILOT_READY.json")
    write_json(output_dir / "implementation_audit_METER_OIA_V1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="preflight")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--real_dino", action="store_true")
    parser.add_argument("--profile_dir", default=".review/meter_oia_v1_real_profile")
    parser.add_argument("--write_review_pass", action="store_true")
    parser.add_argument("--write_pre_pilot_ready", action="store_true")
    parser.add_argument("--test_summary", default="")
    args = parser.parse_args()
    root = Path.cwd()
    report = run_audit(
        root,
        Path(args.output_dir),
        device=args.device,
        real_dino=args.real_dino,
        profile_dir=Path(args.profile_dir),
        write_review_pass=args.write_review_pass,
        write_pre_pilot_ready=args.write_pre_pilot_ready,
        test_summary_path=Path(args.test_summary) if args.test_summary else None,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
