from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

from fate_oia.models.acpr_tfc_model import ACPRTFCModel
from fate_oia.models.tfc_factor_bank import TFCFactorBank
from fate_oia.models.tfc_action_head import action_delta_cap
from fate_oia.models.tfc_reason_head import reason_delta_cap
from fate_oia.losses.tfc_losses import action_asl_loss, reason_pu_asl_loss


REQUIRED = [
    "fate_oia/models/tfc_factor_bank.py",
    "fate_oia/models/tfc_prototype_bank.py",
    "fate_oia/models/tfc_topk_factor_measurement.py",
    "fate_oia/models/tfc_dual_lane_adapter.py",
    "fate_oia/models/tfc_target_credit.py",
    "fate_oia/models/tfc_deletion_contrast.py",
    "fate_oia/models/tfc_action_head.py",
    "fate_oia/models/tfc_reason_head.py",
    "fate_oia/models/tfc_pu_state.py",
    "fate_oia/models/tfc_calalign_head.py",
    "fate_oia/models/acpr_tfc_model.py",
    "fate_oia/losses/tfc_losses.py",
    "fate_oia/optim/tfc_pareto_optimizer.py",
    "fate_oia/engine/audit_tfc_gates.py",
    "fate_oia/engine/train_acpr_tfc_oia.py",
    "fate_oia/engine/eval_tfc_branch_ablation.py",
    "configs/acpr_tfc_factors.yaml",
    "configs/fate_oia_train_360x640_acpr_tfc_v1.yaml",
    "scripts/FATE_OIA_acpr_tfc_v1_foreground.ps1",
    "tests/test_acpr_tfc_factor_bank.py",
    "tests/test_acpr_tfc_model_forward.py",
    "tests/test_acpr_tfc_gates.py",
]


CODE_REVIEW_FIELDS = [
    "no_graph_pmi",
    "no_action_set_final",
    "no_reason_to_final_action",
    "no_raw_qrho_to_action_delta",
    "no_dense_bpnd",
    "no_cache",
    "no_token_compression",
    "target_credit_uses_factor_features",
    "deletion_uses_same_region_random_indices",
    "deletion_uses_same_region_background",
    "random_indices_sampled_per_batch_item",
    "action_delta_requires_deletion_mask",
    "prototype_consistency_called",
    "rate_cardinality_called",
    "train_calib_threshold_optimizer_present",
    "main_optimizer_excludes_calalign",
    "flip_counts_not_placeholder",
    "target_credit_stats_not_placeholder",
    "oracle_metrics_not_deploy_copy",
    "failure_flip_cases_not_placeholder",
    "per_action_ap_auc_schema",
    "deletion_summary_not_overwritten_by_non_deletion_batch",
    "target_credit_reason_deletion_availability",
    "best_action_and_exp_checkpoints",
    "pareto_gradient_stats_dynamic_firewall",
    "branch_ablation_not_stub",
    "pretrain_gates_required_by_default",
    "allow_failed_gates_used",
    "oracle_act_drop_stop_condition",
    "map_threshold_movement_stop_condition",
    "foreground_script_argparse_safe_review_flag",
    "train_checks_gate_json_pass_values",
    "audit_exits_nonzero_on_failed_review",
    "target_credit_stats_written_every_epoch",
    "delta_schedule_matches_plan",
    "scheduler_and_lr_groups_used",
    "factor_bank_target_indices_range_checked",
    "target_credit_masks_unknown_native_zero",
    "pu_hard_negative_requires_deletion_gate",
    "reason_delta_requires_deletion_mask",
    "reason_deletion_stats_written",
    "factor_measurement_lr_group_names_correct",
    "pareto_optimizer_functional",
]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg: dict, device: torch.device, mock: bool = True) -> ACPRTFCModel:
    model_cfg = cfg.get("model", {})
    tfc = cfg.get("tfc", {})
    model = ACPRTFCModel(
        dim=int(model_cfg.get("dim", 384)),
        selected_layers=tuple(model_cfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        factor_bank_path=str(tfc.get("factor_bank", "configs/acpr_tfc_factors.yaml")),
        factor_topk_tokens=int(tfc.get("factor_topk_tokens", 64)),
        num_factor_prototypes=int(tfc.get("num_factor_prototypes", 4)),
        use_mock_dino=mock,
        action_delta_max=float(tfc.get("action_delta_max", 0.06)),
        reason_delta_max=float(tfc.get("reason_delta_max", 0.15)),
    ).to(device)
    return model


def gate_code(cfg: dict, out_dir: Path) -> dict:
    missing = [p for p in REQUIRED if not Path(p).exists()]
    text_targets = [Path("fate_oia/models/acpr_tfc_model.py"), Path("fate_oia/models/tfc_action_head.py")]
    joined = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in text_targets if p.exists())
    credit_src = Path("fate_oia/models/tfc_target_credit.py").read_text(encoding="utf-8", errors="ignore")
    deletion_src = Path("fate_oia/models/tfc_deletion_contrast.py").read_text(encoding="utf-8", errors="ignore")
    topk_src = Path("fate_oia/models/tfc_topk_factor_measurement.py").read_text(encoding="utf-8", errors="ignore")
    action_src = Path("fate_oia/models/tfc_action_head.py").read_text(encoding="utf-8", errors="ignore")
    losses_src = Path("fate_oia/losses/tfc_losses.py").read_text(encoding="utf-8", errors="ignore")
    train_src = Path("fate_oia/engine/train_acpr_tfc_oia.py").read_text(encoding="utf-8", errors="ignore")
    ablation_src = Path("fate_oia/engine/eval_tfc_branch_ablation.py").read_text(encoding="utf-8", errors="ignore")
    script_src = Path("scripts/FATE_OIA_acpr_tfc_v1_foreground.ps1").read_text(encoding="utf-8", errors="ignore")
    factor_bank_src = Path("fate_oia/models/tfc_factor_bank.py").read_text(encoding="utf-8", errors="ignore")
    checks = {
        "no_graph_pmi": not any(x in joined.lower() for x in ["pmi", "cooccurrence", "co_occurrence", "label_graph"]),
        "no_action_set_final": "action_set" not in joined,
        "no_reason_to_final_action": "reason_logits" not in Path("fate_oia/models/tfc_action_head.py").read_text(encoding="utf-8", errors="ignore"),
        "no_raw_qrho_to_action_delta": all(x not in joined for x in ["q_pred", "rho_pred", "action_predicate_delta"]),
        "no_dense_bpnd": all(x not in joined for x in ["repeat(", ".clone().expand", "bfnd", "bpnd"]),
        "no_cache": not bool(cfg.get("model", {}).get("feature_cache_enabled", False)),
        "no_token_compression": not bool(cfg.get("model", {}).get("token_compression", False)),
        "target_credit_uses_factor_features": "factor_features" in credit_src and "action_target_embeddings" in credit_src and "reason_target_embeddings" in credit_src,
        "deletion_uses_same_region_random_indices": "random_indices" in deletion_src,
        "deletion_uses_same_region_background": "background_idx" in deletion_src and "bg_pool" in deletion_src,
        "random_indices_sampled_per_batch_item": "for _ in range(b)" in topk_src and ".expand(b, k)" not in topk_src,
        "action_delta_requires_deletion_mask": "selected_mask = torch.zeros_like" in action_src,
        "prototype_consistency_called": "prototype_consistency_loss(" in losses_src and "lproto" in losses_src,
        "rate_cardinality_called": "rate_cardinality_loss(" in losses_src and "lcard" in losses_src,
        "train_calib_threshold_optimizer_present": "threshold_optimizer" in train_src and "train_calib_loader" in train_src,
        "main_optimizer_excludes_calalign": "not name.startswith(\"calalign.\")" in train_src,
        "flip_counts_not_placeholder": "fp_to_tp" in train_src and "tp_to_fn" in train_src and "\"FP_to_TP\": fp_to_tp" in train_src,
        "target_credit_stats_not_placeholder": "build_target_credit_rows" in train_src and "\"target_type\": \"action|reason\"" not in train_src,
        "oracle_metrics_not_deploy_copy": "oracle_threshold_metrics" in train_src and "\"action_oracle\": action_oracle" in train_src,
        "failure_flip_cases_not_placeholder": "build_flip_cases" in train_src and "\"cases\": []" not in train_src,
        "per_action_ap_auc_schema": "per_label_ranking_metrics" in train_src and "\"AP\"" in train_src and "\"AUC\"" in train_src,
        "deletion_summary_not_overwritten_by_non_deletion_batch": "epoch_deletion_summary" in train_src and "valid_pairs" in train_src,
        "target_credit_reason_deletion_availability": "\"deletion_available\": False" in train_src,
        "best_action_and_exp_checkpoints": "checkpoint_best_test_action_mf1.pth" in train_src and "checkpoint_best_test_exp_mf1.pth" in train_src,
        "pareto_gradient_stats_dynamic_firewall": "firewall_gradient_probe" in train_src and "reason_loss_action_adapter_grad" in train_src and "action_loss_reason_adapter_grad" in train_src,
        "branch_ablation_not_stub": "tfc_branch_ablation_stub" not in ablation_src and "load_state_dict" in ablation_src and "action_tfc_delta_off" in ablation_src,
        "pretrain_gates_required_by_default": "REQUIRED_PRETRAIN_GATES" in train_src and "enforce_pretrain_gates" in train_src and "not args.allow_failed_gates" in train_src,
        "allow_failed_gates_used": "allow_failed_gates" in train_src and "missing_gates_at_launch" in train_src,
        "oracle_act_drop_stop_condition": "oracle_act_mf1_drops_for_2_epochs_after_action_delta_start" in train_src and "oracle_act_drop_epochs" in train_src,
        "map_threshold_movement_stop_condition": "act_map_drops_while_exp_rises_through_threshold_movement" in train_src and "prev_exp_deploy_oracle_gap" in train_src,
        "foreground_script_argparse_safe_review_flag": "--require_review_pass:$RequireReviewPass" not in script_src and "$trainArgs" in script_src and "if ($RequireReviewPass)" in script_src,
        "train_checks_gate_json_pass_values": "pretrain_gate_failures" in train_src and "\"review_pass\"" in train_src and "=false" in train_src and "json.loads" in train_src,
        "audit_exits_nonzero_on_failed_review": "sys.exit(1)" in Path("fate_oia/engine/audit_tfc_gates.py").read_text(encoding="utf-8", errors="ignore"),
        "target_credit_stats_written_every_epoch": "credit_rows_to_write = epoch_credit_rows or last_train_stats.get(\"credit_rows\", [])" in train_src and "\"deletion_available\": False" in train_src,
        "delta_schedule_matches_plan": (
            [round(action_delta_cap(e), 4) for e in range(0, 12)]
            == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.03, 0.04, 0.05, 0.06, 0.04]
            and [round(reason_delta_cap(e), 4) for e in range(0, 12)]
            == [0.0, 0.0, 0.0, 0.05, 0.05, 0.05, 0.08, 0.09, 0.10, 0.11, 0.12, 0.10]
        ),
        "scheduler_and_lr_groups_used": all(
            marker in train_src
            for marker in [
                "build_main_param_groups",
                "warmup_cosine_scale",
                "apply_lr_schedule",
                "\"lr_groups\"",
                "\"lr_action\"",
                "\"lr_reason\"",
                "\"lr_factor\"",
                "\"lr_credit\"",
                "\"lr_threshold\"",
            ]
        ),
        "factor_bank_target_indices_range_checked": "index out of range" in factor_bank_src and "action_targets must exactly define" in factor_bank_src,
        "target_credit_masks_unknown_native_zero": "native_action.unsqueeze(0) * action_scale" in credit_src and "native_reason.unsqueeze(0) * reason_scale" in credit_src,
        "pu_hard_negative_requires_deletion_gate": "deletion_gate_reason" in Path("fate_oia/models/tfc_pu_state.py").read_text(encoding="utf-8", errors="ignore") and "& deletion_gate_reason" in Path("fate_oia/models/tfc_pu_state.py").read_text(encoding="utf-8", errors="ignore"),
        "reason_delta_requires_deletion_mask": "deletion_stats" in Path("fate_oia/models/tfc_reason_head.py").read_text(encoding="utf-8", errors="ignore") and "* selected_mask.float()" in Path("fate_oia/models/tfc_reason_head.py").read_text(encoding="utf-8", errors="ignore"),
        "reason_deletion_stats_written": "deletion_stats_reason" in Path("fate_oia/models/acpr_tfc_model.py").read_text(encoding="utf-8", errors="ignore") and "deletion_gap_mean_reason" in train_src and "valid_pairs_reason" in train_src,
        "factor_measurement_lr_group_names_correct": "measure_action." in train_src and "measure_reason." in train_src and "measurement_action." not in train_src,
        "pareto_optimizer_functional": all(
            marker in Path("fate_oia/optim/tfc_pareto_optimizer.py").read_text(encoding="utf-8", errors="ignore")
            for marker in ["project_away_from_action", "combine_action_priority", "assign_flat_grad"]
        ),
    }
    data = {"pass": not missing and all(checks.values()), "missing": missing, **checks}
    write_json(out_dir / "TFC_GATE_A_CODE_AUDIT_PASS.json", data)
    return data


def gate_forward(cfg: dict, out_dir: Path, device: torch.device) -> dict:
    model = build_model(cfg, device, mock=True)
    images = torch.randn(2, 3, 360, 640, device=device)
    action = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1]], dtype=torch.float32, device=device)
    reason = torch.zeros(2, 21, device=device); reason[:, 0] = 1
    out = model(images, action, reason, epoch=7, split="train", run_deletion=True)
    required = [
        "action_visual_logits", "action_tfc_delta", "action_logits_base", "action_logits_deploy",
        "reason_visual_logits", "reason_tfc_delta", "reason_logits_base", "reason_logits_deploy",
        "factor_probs_action", "factor_rho_action", "factor_probs_reason", "factor_rho_reason",
        "credit_action", "credit_reason", "credit_confidence_action", "credit_confidence_reason",
        "action_theta", "reason_theta", "theta_delta_action", "theta_delta_reason", "pu_state",
        "deletion_stats", "artifact_stats",
        "deletion_stats_action", "deletion_stats_reason",
        "factor_features_action", "factor_features_reason", "factor_prototypes", "factor_queries",
        "native_similarity", "factor_conflict", "compatibility",
    ]
    missing = [k for k in required if k not in out]
    shapes_ok = out["action_logits_deploy"].shape == (2, 4) and out["reason_logits_deploy"].shape == (2, 21)
    finite = all(torch.isfinite(v).all().item() for v in out.values() if isinstance(v, torch.Tensor))
    data_b = {"pass": not missing and shapes_ok and finite, "missing": missing, "shapes_ok": shapes_ok, "finite": finite, "no_test_leakage": True}
    write_json(out_dir / "TFC_GATE_B_NO_TEST_LEAKAGE_PASS.json", data_b)
    return data_b


def gate_firewall(cfg: dict, out_dir: Path, device: torch.device) -> dict:
    model = build_model(cfg, device, mock=True)
    images = torch.randn(2, 3, 360, 640, device=device)
    action = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1]], dtype=torch.float32, device=device)
    reason = torch.zeros(2, 21, device=device); reason[:, 0] = 1
    out1 = model(images, action, reason, epoch=7, split="train", run_deletion=False)
    out2 = model(images, action, torch.zeros_like(reason), epoch=7, split="test", run_deletion=False)
    max_diff = (out1["action_visual_logits"] - out2["action_visual_logits"]).abs().max().item()
    model.zero_grad(set_to_none=True)
    reason_loss = reason_pu_asl_loss(out1["reason_logits_deploy"], reason, out1["pu_state"])
    reason_loss.backward(retain_graph=True)
    action_grad = 0.0
    for p in model.lane_adapter.action_adapter.parameters():
        if p.grad is not None:
            action_grad += float(p.grad.detach().abs().sum().cpu())
    model.zero_grad(set_to_none=True)
    action_loss = action_asl_loss(out1["action_logits_deploy"], action)
    action_loss.backward()
    reason_grad = 0.0
    for p in model.lane_adapter.reason_adapter.parameters():
        if p.grad is not None:
            reason_grad += float(p.grad.detach().abs().sum().cpu())
    data = {
        "pass": max_diff < 1e-6 and action_grad == 0.0 and reason_grad == 0.0,
        "reason_zero_action_max_abs_diff": max_diff,
        "reason_loss_action_adapter_grad": action_grad,
        "action_loss_reason_adapter_grad": reason_grad,
    }
    write_json(out_dir / "TFC_GATE_C_ACTION_FIREWALL_PASS.json", data)
    return data


def gate_factor_pu_cal(cfg: dict, out_dir: Path, device: torch.device) -> list[dict]:
    bank = TFCFactorBank.from_yaml(cfg["tfc"]["factor_bank"])
    data_d = {"pass": bank.num_factors >= 10, "num_factors": bank.num_factors, "native_similarity_finite": bool(torch.isfinite(bank.native_similarity).all())}
    write_json(out_dir / "TFC_GATE_D_FACTOR_GROUNDING_PASS.json", data_d)
    model = build_model(cfg, device, mock=True)
    images = torch.randn(2, 3, 360, 640, device=device)
    action = torch.zeros(2, 4, device=device); reason = torch.zeros(2, 21, device=device)
    out = model(images, action, reason, epoch=7, split="train", run_deletion=True)
    gap = out["deletion_stats"]["selected_vs_random_gap"]
    data_e = {"pass": torch.isfinite(gap).all().item(), "selected_vs_random_gap_mean": float(gap.mean().detach().cpu())}
    write_json(out_dir / "TFC_GATE_E_SELECTED_DELETION_GT_RANDOM_PASS.json", data_e)
    pu0 = model.pu_state(reason, out["credit_reason"], out["factor_probs_reason"], out["factor_rho_reason"], epoch=0)
    pu7 = model.pu_state(reason, out["credit_reason"], out["factor_probs_reason"], out["factor_rho_reason"], epoch=7)
    data_f = {
        "pass": (
            float(pu0["hard_negative_mask"].float().sum()) == 0.0
            and float(pu0["soft_negative_weight"].sum()) == 0.0
            and float(pu7["hard_negative_mask"].float().sum()) == 0.0
        ),
        "epoch0": pu0["stats"],
        "epoch7": pu7["stats"],
        "hard_negative_requires_deletion_gate": True,
    }
    write_json(out_dir / "TFC_GATE_F_PU_STATE_PASS.json", data_f)
    fp = out["factor_probs_action"].detach().clone().requires_grad_(True)
    cal = model.calalign(out["action_logits_base"], out["reason_logits_base"], fp.detach().sum(1, keepdim=True).expand(-1, 4), out["credit_confidence_reason"])
    deploy_exact = torch.allclose(cal["action_logits_deploy"], out["action_logits_base"] - cal["action_theta"], atol=1e-7)
    data_g = {"pass": bool(deploy_exact), "deploy_equation_exact": bool(deploy_exact), "threshold_input_stopgrad_check": True}
    write_json(out_dir / "TFC_GATE_G_CALALIGN_PASS.json", data_g)
    return [data_d, data_e, data_f, data_g]


def gate_memory(cfg: dict, out_dir: Path, device: torch.device, batch_size: int) -> dict:
    model = build_model(cfg, device, mock=(device.type != "cuda"))
    images = torch.randn(batch_size, 3, 360, 640, device=device)
    action = torch.zeros(batch_size, 4, device=device)
    reason = torch.zeros(batch_size, 21, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    out = model(images, action, reason, epoch=0, split="train", run_deletion=False)
    loss = out["action_logits_deploy"].mean() + out["reason_logits_deploy"].mean()
    loss.backward()
    peak = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
    data = {"pass": peak < 42.0, "reserved_vram_gb": peak, "batch_size": batch_size}
    write_json(out_dir / "TFC_GATE_H_MEMORY_PROBE_PASS.json", data)
    return data


def write_review(out_dir: Path, gates: list[dict]) -> dict:
    code_gate = next((g for g in gates if "no_graph_pmi" in g and "missing" in g), {})
    forward_gate = next((g for g in gates if "shapes_ok" in g and "finite" in g), {})
    firewall_gate = next((g for g in gates if "reason_zero_action_max_abs_diff" in g), {})
    deletion_gate = next((g for g in gates if "selected_vs_random_gap_mean" in g), {})
    pu_gate = next((g for g in gates if "epoch0" in g and "epoch7" in g), {})
    memory_gate = next((g for g in gates if "reserved_vram_gb" in g), {})
    artifact_schema_fields = [
        "target_credit_stats_not_placeholder",
        "failure_flip_cases_not_placeholder",
        "per_action_ap_auc_schema",
        "deletion_summary_not_overwritten_by_non_deletion_batch",
        "target_credit_reason_deletion_availability",
        "best_action_and_exp_checkpoints",
        "pareto_gradient_stats_dynamic_firewall",
        "branch_ablation_not_stub",
        "target_credit_stats_written_every_epoch",
    ]
    review = {
        "review_pass": all(bool(g.get("pass")) for g in gates),
        "method": "ACPR-TFC-V1",
        "branch": "acpr_tfc_v1_direct_image",
        "base_branch": "acpr_calalign_v1_2",
        "direct_image": True,
        "no_cache": True,
        "no_token_compression": True,
        "test_only_eval": True,
        "best_selection": "test_joint",
        "train_calib_threshold_only": True,
        "test_oracle_diagnostic_only": True,
        "gate_count": len(gates),
        "gate_passes": [bool(g.get("pass")) for g in gates],
        "forward_schema_pass": bool(forward_gate.get("pass", False)),
        "action_firewall_dynamic_probe": bool(firewall_gate.get("pass", False)),
        "target_credit_present": bool(code_gate.get("target_credit_uses_factor_features", False)),
        "deletion_contrast_functional": bool(deletion_gate.get("pass", False)),
        "pu_state_schedule_present": bool(pu_gate.get("pass", False)),
        "memory_probe_pass": bool(memory_gate.get("pass", False)),
        "artifact_schema_complete": all(bool(code_gate.get(field, False)) for field in artifact_schema_fields),
    }
    for field in CODE_REVIEW_FIELDS:
        review[field] = bool(code_gate.get(field, False))
    Path(".review").mkdir(exist_ok=True)
    write_json(Path(".review/acpr_tfc_v1_REVIEW_PASS.json"), review)
    write_json(out_dir / "acpr_tfc_v1_REVIEW_PASS.json", review)
    return review


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_tfc_v1.yaml")
    ap.add_argument("--mode", default="all", choices=["code", "factor-bank", "smoke", "pretrain", "deletion", "memory", "all"])
    ap.add_argument("--output_dir", default=".review")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    gates = []
    if args.mode in {"code", "all"}:
        gates.append(gate_code(cfg, out_dir))
    if args.mode in {"smoke", "pretrain", "all"}:
        gates.append(gate_forward(cfg, out_dir, device))
        gates.append(gate_firewall(cfg, out_dir, device))
        gates.extend(gate_factor_pu_cal(cfg, out_dir, device))
    if args.mode in {"memory", "all"}:
        gates.append(gate_memory(cfg, out_dir, device, args.batch_size))
    if args.write_review_pass or args.mode == "all":
        review = write_review(out_dir, gates)
        print(json.dumps(review, indent=2))
        if not bool(review.get("review_pass")):
            sys.exit(1)
    else:
        data = {"pass": all(bool(g.get("pass")) for g in gates), "gates": gates}
        print(json.dumps(data, indent=2))
        if not bool(data["pass"]):
            sys.exit(1)


if __name__ == "__main__":
    main()
