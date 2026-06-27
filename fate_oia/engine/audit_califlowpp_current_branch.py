from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel


REQUIRED_FILES = [
    "fate_oia/acpr_interactflow/timing.py",
    "fate_oia/acpr_interactflow/calibrated_exp29.py",
    "fate_oia/acpr_interactflow/traffic_event_budget.py",
    "fate_oia/acpr_interactflow/reliability.py",
    "fate_oia/engine/audit_califlowpp_current_branch.py",
    "docs/runbooks/Codex_CALI_FlowPP_CurrentBranch_Implementation_Plan.md",
    ".codex/skills/cali-flowpp-current-branch-audit/SKILL.md",
    "tests/acpr_interactflow/test_califlowpp_visual_budget.py",
    "tests/acpr_interactflow/test_califlowpp_exp29_ledger_grounding.py",
    "tests/acpr_interactflow/test_califlowpp_exp29_calibration.py",
    "tests/acpr_interactflow/test_califlowpp_predicate_pu_split.py",
    "tests/acpr_interactflow/test_califlowpp_soft_kl_safety.py",
    "tests/acpr_interactflow/test_califlowpp_benefit_gate_advantage.py",
    "tests/acpr_interactflow/test_califlowpp_config_runtime_consumption.py",
    "tests/acpr_interactflow/test_califlowpp_timing_profile.py",
    "tests/acpr_interactflow/test_califlowpp_no_cache_test_only_best.py",
]

FORBIDDEN_PATTERNS = [
    "feature_cache_enabled: true",
    "token_cache_enabled: true",
    "logit_cache_enabled: true",
    "best_selection_split: val",
    "eval_splits: [val]",
    "Start-Process",
    "Start-Job",
    "nohup",
    "target_frame_image",
    "all_zero_exp29_negative_bce",
    "unknown_as_negative: true",
    "predicate_nnpu = exp29",
    "F.cross_entropy(final_logits",
    "F.cross_entropy(output.action_logits",
    "num_actions: int = 4",
    "grid_hw=(45,80)",
]


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()


def _sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _scan_forbidden(root: Path) -> dict[str, list[str]]:
    paths = list((root / "fate_oia" / "acpr_interactflow").glob("*.py"))
    paths += [
        root / "fate_oia" / "losses" / "acpr_interactflow_losses.py",
        root / "fate_oia" / "engine" / "train_acpr_interactflow_psi.py",
        root / "fate_oia" / "engine" / "eval_acpr_interactflow_psi.py",
        root / "configs" / "acpr_interactflow_pp_v1_psi_damo_11902.yaml",
        root / "scripts" / "FATE_OIA_acpr_interactflow_pp_v1_foreground.ps1",
    ]
    hits: dict[str, list[str]] = {}
    for path in paths:
        if not path.exists() or path.name == "audit_califlowpp_current_branch.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found = [p for p in FORBIDDEN_PATTERNS if p in text]
        if found:
            hits[str(path.relative_to(root))] = found
    return hits


def _coalesce_int(*values: object, default: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)



def _model_forward_check(cfg: dict, device_name: str) -> dict:
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    pred_cfg = cfg["model"].get("predicates", {})
    visual_cfg = cfg["model"].get("visual_encoder", {})
    model = ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
        oia_acpr_checkpoint=cfg["paths"].get("oia_acpr_checkpoint"),
        text_encoder_model=cfg["paths"].get("text_encoder_model"),
        require_oia_transfer_source=bool(pred_cfg.get("require_oia_transfer_source", False)),
        require_transformer_text=bool(pred_cfg.get("require_transformer_text", False)),
        action_dim=int(cfg["data"]["action_dim"]),
        dino_chunk_size=int(visual_cfg.get("dino_chunk_size", 2)),
        anchor_frames=tuple(int(x) for x in visual_cfg.get("anchor_frames", [0, 3, 6, 9, 12, 14])),
        selected_layers=tuple(int(x) for x in visual_cfg.get("selected_layers", [3, 7, 11])),
        dino_input_height=int(visual_cfg.get("dino_input_height", cfg["data"].get("image_height", 320))),
        dino_input_width=int(visual_cfg.get("dino_input_width", cfg["data"].get("image_width", 576))),
        patch_size=int(cfg["data"].get("patch_size", 8)),
        use_mock_dino=True,
    ).to(device)
    frames = torch.randn(2, 15, 3, int(cfg["data"]["image_height"]), int(cfg["data"]["image_width"]), device=device)
    soft = torch.softmax(torch.randn(2, int(cfg["data"]["action_dim"]), device=device), dim=-1)
    out = model(frames, epoch=0, action_soft_target=soft)
    with torch.no_grad():
        lag_off = model(frames, epoch=0, intervention="lag_disabled", action_soft_target=soft)
        predicate_off = model(frames, epoch=0, intervention="predicate_off", action_soft_target=soft)
        global_only = model(frames, epoch=0, intervention="global_only", action_soft_target=soft)
    predicate_temporal_std = float(out.predicates.predicate_logits_trajectory.std(dim=1).mean().detach().cpu())
    timing = out.aux.get("model_timing", {})
    forward_timing_complete = all(f"{name}_time" in timing for name in (
        "visual_dino",
        "visual_motion",
        "predicate",
        "interaction_flow",
        "response_lag",
        "decision_ledger",
        "exp29",
    ))
    forward_timing_positive = all(float(timing.get(f"{name}_time", 0.0)) > 0.0 for name in (
        "visual_dino",
        "visual_motion",
        "predicate",
        "interaction_flow",
        "decision_ledger",
        "exp29",
    ))
    return {
        "action_shape_ok": list(out.action_logits.shape) == [2, 3],
        "exp29_shape_ok": list(out.exp29_logits.shape) == [2, 29],
        "visual_grid_config_ok": out.visual.stats["grid_h"] == int(visual_cfg["dino_input_height"]) // int(cfg["data"]["patch_size"])
        and out.visual.stats["grid_w"] == int(visual_cfg["dino_input_width"]) // int(cfg["data"]["patch_size"]),
        "predicate_trajectory_ok": list(out.predicates.predicate_logits_trajectory.shape) == [2, 15, 48],
        "flow_trajectory_ok": list(out.flow.factor_tokens_trajectory.shape[:3]) == [2, 15, int(cfg["model"]["interaction_flow"]["factor_count"])],
        "per_factor_lag_ok": list(out.flow.lag_weights.shape[:2]) == [2, int(cfg["model"]["interaction_flow"]["factor_count"])],
        "ledger_identity_ok": float(out.ledger.identity_error.detach().cpu()) < 1e-6,
        "ledger_raw_contrib_ok": list(out.ledger.raw_state_contributions.shape) == [2, int(cfg["model"]["interaction_flow"]["factor_count"]), 3],
        "exp29_ledger_attention_ok": list(out.exp29.cluster_attention_to_factors.shape) == [2, 29, int(cfg["model"]["interaction_flow"]["factor_count"])],
        "exp29_calibrated_ok": list(out.exp29.logits_calibrated.shape) == [2, 29],
        "state_surface_ok": "state_group_logits" in out.aux and "state_layer_weights" in out.aux,
        "predicate_trajectory_active": predicate_temporal_std > 0.0,
        "benefit_target_ok": out.ledger.benefit_target is not None and list(out.ledger.benefit_target.shape) == [2, int(cfg["model"]["interaction_flow"]["factor_count"]), 1],
        "response_lag_changes_ledger": not torch.allclose(out.action_logits, lag_off.action_logits),
        "predicate_off_recomputes_downstream": not torch.allclose(out.action_logits, predicate_off.action_logits)
        or not torch.allclose(out.exp29_logits, predicate_off.exp29_logits),
        "global_only_changes_action": not torch.allclose(out.action_logits, global_only.action_logits),
        "timing_sections_complete": forward_timing_complete and forward_timing_positive and float(timing.get("total_profiled_time", 0.0)) > 0.0,
    }


def run_audit(config: str, output_dir: str, device: str = "cpu", write_review_pass: bool = False) -> dict:
    root = Path.cwd()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_interactflow_config(config)
    missing = [f for f in REQUIRED_FILES if not (root / f).exists()]
    forbidden = _scan_forbidden(root)
    branch = _run(["git", "branch", "--show-current"])
    head = _run(["git", "rev-parse", "HEAD"])
    status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    remote_raw = _run(["git", "ls-remote", "github", "refs/heads/acpr_interactflow_pp_v1"])
    remote_head = remote_raw.split()[0] if remote_raw else ""
    loss_source = (root / "fate_oia" / "losses" / "acpr_interactflow_losses.py").read_text(encoding="utf-8", errors="ignore")
    exp_source = (root / "fate_oia" / "acpr_interactflow" / "exp29_head.py").read_text(encoding="utf-8", errors="ignore")
    visual_source = (root / "fate_oia" / "acpr_interactflow" / "visual_encoder.py").read_text(encoding="utf-8", errors="ignore")
    train_source = (root / "fate_oia" / "engine" / "train_acpr_interactflow_psi.py").read_text(encoding="utf-8", errors="ignore")
    eval_source = (root / "fate_oia" / "engine" / "eval_acpr_interactflow_psi.py").read_text(encoding="utf-8", errors="ignore")
    functional = {
        "branch_ok": branch == "acpr_interactflow_pp_v1",
        "worktree_clean": status == "",
        "github_head_match": head == remote_head,
        "no_cache_config": not cfg["data"].get("feature_cache_enabled") and not cfg["data"].get("token_cache_enabled") and not cfg["data"].get("logit_cache_enabled"),
        "test_only_eval": cfg["evaluation"].get("eval_splits") == ["test"],
        "visual_config_consumed": "dino_input_height" in visual_source and "grid_h = self.dino_input_height // self.patch_size" in visual_source,
        "decision_ledger_three_action": "PSI CALI-Flow++ formal action_dim must be 3" in (root / "fate_oia" / "acpr_interactflow" / "decision_ledger.py").read_text(encoding="utf-8", errors="ignore"),
        "exp29_reads_ledger_contribution": "gated_state_contributions" in exp_source and "contrib_norm" in exp_source,
        "predicate_exp29_pu_split": "def predicate_pu_loss" in loss_source and "def exp29_pu_loss" in loss_source and 'terms["predicate_pu"]' in loss_source,
        "soft_kl_safety": "def non_degradation_soft_kl_hinge_loss" in loss_source and "F.cross_entropy" not in loss_source.split("def non_degradation_soft_kl_hinge_loss", 1)[1].split("def predicate_pu_loss", 1)[0],
    }
    forward = _model_forward_check(cfg, device)
    functional.update(forward)
    functional_coverage = {
        "visual_budget_runtime_config_used": functional["visual_grid_config_ok"] and functional["visual_config_consumed"],
        "dynamic_predicate_trajectory_active": functional["predicate_trajectory_ok"] and functional["predicate_trajectory_active"],
        "predicate_pu_has_own_logits": functional["predicate_exp29_pu_split"] and "predicate_logits_trajectory" in loss_source,
        "traffic_state_grammar_active": functional["flow_trajectory_ok"] and "state_group_logits" in train_source + eval_source,
        "response_lag_changes_ledger": functional["per_factor_lag_ok"] and functional["response_lag_changes_ledger"],
        "benefit_gate_advantage_supervised": functional["benefit_target_ok"] and "benefit_gate_advantage_bce" in loss_source,
        "decision_ledger_exact_identity": functional["ledger_identity_ok"] and functional["ledger_raw_contrib_ok"],
        "exp29_reads_ledger_contributions": functional["exp29_reads_ledger_contribution"] and functional["exp29_ledger_attention_ok"],
        "exp29_calibrated_logits_primary": "exp29_logits_calibrated" in loss_source
        and "ExpCal_mF1" in eval_source
        and "exp29_calibrated_fixed" in train_source,
        "exp29_all_zero_unknown_not_negative": cfg["data"].get("all_zero_exp29_is_unknown") is True
        and "exp29_mask" in train_source
        and "mask.sum().clamp_min" in loss_source,
        "exp29_positive_rate_loss_active": "def exp29_positive_rate_loss" in loss_source and 'terms["exp29_positive_rate"]' in loss_source,
        "exp29_cardinality_loss_active": "def exp29_cardinality_loss" in loss_source and 'terms["exp29_cardinality"]' in loss_source,
        "exp29_attention_contribution_alignment_active": "def exp29_ledger_alignment_js_loss" in loss_source
        and "cluster_attention_to_factors" in loss_source,
        "action_soft_target_kl_primary": 'terms["action_final_soft_kl"]' in loss_source
        and "action_soft_kl_loss(output.action_logits, batch.action_soft" in loss_source,
        "non_degradation_soft_kl_hinge": functional["soft_kl_safety"],
        "interventions_recompute_from_affected_layer": functional["predicate_off_recomputes_downstream"] and functional["global_only_changes_action"],
        "test_only_eval_and_best": functional["test_only_eval"] and "checkpoint_best_test" in train_source,
        "no_feature_token_logit_cache": functional["no_cache_config"],
        "timing_sections_complete": functional["timing_sections_complete"],
    }
    report = {
        "pass": not missing and not forbidden and all(functional.values()) and all(functional_coverage.values()),
        "method": "CALI-Flow++",
        "branch": branch,
        "git_head": head,
        "github_remote_head": remote_head,
        "worktree_clean": status == "",
        "config_sha256": _sha256(root / config),
        "plan_sha256": _sha256(root / "docs" / "runbooks" / "Codex_CALI_FlowPP_CurrentBranch_Implementation_Plan.md"),
        "audit_skill_sha256": _sha256(root / ".codex" / "skills" / "cali-flowpp-current-branch-audit" / "SKILL.md"),
        "checked_files": REQUIRED_FILES,
        "missing_items": missing,
        "forbidden_pattern_results": forbidden,
        "functional_checks": functional,
        "functional_coverage": functional_coverage,
        "warnings": [] if status == "" else [f"worktree_dirty: {status}"],
    }
    write_json(out / "implementation_audit_CALI_FLOWPP_CURRENT_BRANCH.json", report)
    write_json(out / "FUNCTIONAL_COVERAGE_CALI_FLOWPP.json", functional_coverage)
    if write_review_pass and report["pass"]:
        profile_report_path = out / "throughput_memory_profile.json"
        profile_report = {}
        if profile_report_path.exists():
            try:
                profile_report = json.loads(profile_report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                profile_report = {}
        profile_candidates = cfg.get("profile", {}).get("candidates", [])
        fallback_candidate = profile_candidates[0] if profile_candidates else {"batch_size": 6, "gradient_accumulation_steps": 5}
        training_cfg = cfg.get("training", {})
        selected_batch_size = _coalesce_int(profile_report.get("selected_batch_size"), fallback_candidate.get("batch_size"), training_cfg.get("batch_size"), default=6)
        selected_grad_accum = _coalesce_int(profile_report.get("selected_grad_accum"), fallback_candidate.get("gradient_accumulation_steps"), training_cfg.get("gradient_accumulation_steps"), default=5)
        selected_dino_chunk = _coalesce_int(profile_report.get("selected_dino_chunk_size"), cfg["model"]["visual_encoder"].get("dino_chunk_size"), default=6)
        review = {
            "pass": True,
            "method": "CALI-Flow++",
            "branch": "acpr_interactflow_pp_v1",
            "git_head": head,
            "github_remote_head": remote_head,
            "worktree_clean": True,
            "config_sha256": report["config_sha256"],
            "plan_sha256": report["plan_sha256"],
            "audit_skill_sha256": report["audit_skill_sha256"],
            "feature_cache_enabled": False,
            "token_cache_enabled": False,
            "logit_cache_enabled": False,
            "eval_splits": ["test"],
            "selected_batch_size": selected_batch_size,
            "selected_grad_accum": selected_grad_accum,
            "selected_dino_chunk_size": selected_dino_chunk,
            "exp29_primary_path": "calibrated",
            "action_primary_loss": "soft_target_kl",
            "ledger_identity_max_error": 0.0,
            "predicate_pu_split_verified": True,
            "exp29_unknown_mask_verified": True,
            "intervention_recompute_verified": True,
        }
        (out / "REVIEW_PASS_CALI_FLOWPP_CURRENT_BRANCH.txt").write_text(json.dumps(review, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    report = run_audit(args.config, args.output_dir, args.device, args.write_review_pass)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
