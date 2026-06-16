from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml


REQUIRED = [
    "fate_oia/models/acpr_dino_field.py",
    "fate_oia/models/acpr_ego_regions.py",
    "fate_oia/models/acpr_scene_predicate_head.py",
    "fate_oia/models/acpr_predicate_targets.py",
    "fate_oia/models/acpr_pair_memory.py",
    "fate_oia/models/acpr_action_combo_aux.py",
    "fate_oia/models/acpr_calibration.py",
    "fate_oia/losses/acpr_losses.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_predicate_reason.py",
    "fate_oia/engine/supervise_acpr_oia_foreground.py",
    "fate_oia/engine/train_acpr_oia.py",
]

CALALIGN_REQUIRED = [
    "fate_oia/models/acpr_threshold_head.py",
    "fate_oia/losses/acpr_threshold_losses.py",
    "fate_oia/utils/acpr_threshold_search.py",
    "fate_oia/utils/acpr_train_calib_split.py",
    "fate_oia/engine/fit_acpr_threshold_head.py",
    "configs/fate_oia_train_360x640_acpr_calalign_v1_2.yaml",
    "scripts/FATE_OIA_acpr_calalign_v1_2_foreground.ps1",
]

ACTALIGN_REQUIRED = [
    "fate_oia/models/acpr_action_utility.py",
    "fate_oia/models/acpr_action_predicate_delta.py",
    "fate_oia/losses/acpr_action_utility_losses.py",
    "fate_oia/utils/acpr_action_pareto_gate.py",
    "fate_oia/utils/acpr_action_gradient_guard.py",
    "fate_oia/utils/acpr_model_ema.py",
    "fate_oia/utils/acpr_swa_lite.py",
    "configs/fate_oia_train_360x640_acpr_actalign_v1_3.yaml",
    "scripts/FATE_OIA_acpr_actalign_v1_3_foreground.ps1",
]

ACTALIGN_CANDIDATE_REQUIRED = [
    "fate_oia/models/acpr_action_candidates.py",
    "fate_oia/losses/acpr_candidate_losses.py",
    "fate_oia/utils/acpr_candidate_gate.py",
    "fate_oia/utils/acpr_candidate_metrics.py",
    "fate_oia/engine/fit_acpr_action_candidates.py",
    "configs/fate_oia_train_360x640_acpr_actalign_v1_3_candidate_probe.yaml",
]


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    calalign_enabled = bool(cfg.get("threshold", {}).get("enabled", False))
    actalign_enabled = bool(cfg.get("actalign", {}).get("enabled", False) or cfg.get("model", {}).get("actalign_enabled", False))
    candidate_probe = str(cfg.get("actalign", {}).get("stage_mode", cfg.get("actalign", {}).get("mode", ""))) == "candidate_probe"
    required_files = REQUIRED + (CALALIGN_REQUIRED if calalign_enabled else []) + (ACTALIGN_REQUIRED if actalign_enabled else []) + (ACTALIGN_CANDIDATE_REQUIRED if candidate_probe else [])
    missing: list[str] = []
    checks: dict[str, bool] = {}
    for rel in required_files:
        p = Path(rel)
        if not p.exists():
            missing.append(rel); continue
        if p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8"))
    checks["test_only_no_cache_no_compression"] = cfg.get("eval_splits") == "test" and cfg.get("best_selection_split") == "test" and cfg.get("token_compression") == "none" and cfg.get("feature_cache_enabled") is False
    grammar = yaml.safe_load(Path("configs/acpr_reason_predicate_grammar.yaml").read_text(encoding="utf-8")) or {}
    display = yaml.safe_load(Path("configs/bdd_oia_reason_names_external.yaml").read_text(encoding="utf-8")) or {}
    checks["grammar_matches_external_names"] = all(str(grammar["reasons"][i]["name"]) == str(display["names"][i]) for i in range(21))
    checks["grammar_has_predicate_fields"] = all(all(k in grammar["reasons"][i] for k in ["positive_predicates", "contradictory_predicates", "compatible_actions", "hard_negative_reasons", "spatial_region"]) for i in range(21))
    text = "\n".join(Path(r).read_text(encoding="utf-8") for r in required_files if Path(r).exists())
    checks["action_visual_head_per_action_token"] = all(s in Path("fate_oia/models/acpr_label_trunk.py").read_text(encoding="utf-8") for s in ["action_visual_head(action_nodes).squeeze(-1)", "action_token_norm_mean"]) and "action_visual(label_nodes[:, : self.action_dim].mean(1))" not in Path("fate_oia/models/acpr_label_trunk.py").read_text(encoding="utf-8")
    checks["pair_mining_reason_specific"] = all(s in text for s in ["pair_reason_ids", "pair_pos_indices", "pair_neg_indices", "pair_contradiction", "global_embedding", "predicate_probs"])
    checks["hardpair_active_schema"] = all(s in Path("fate_oia/models/acpr_pair_memory.py").read_text(encoding="utf-8") for s in ["pair_active_mask", "pair_hard_mask", "pair_semi_hard_mask", "pair_easy_mask", "pair_neg_logits_detached", "pair_neg_embedding_detached"])
    checks["pair_loss_reason_specific"] = "reason_logits[pos, rid]" in text and "margin - z_pos + z_neg" in text
    checks["memory_pair_loss_detached"] = all(s in Path("fate_oia/losses/acpr_losses.py").read_text(encoding="utf-8") for s in ["pair_neg_is_memory", "pair_neg_logits_detached", "pair_neg_embedding_detached", "detach()"])
    checks["pu_loss_uses_contradiction"] = "0.2 + (1.0 - neg_min_weight) * contradiction_scores" in text or "neg_min_weight + (1.0 - neg_min_weight) * contradiction_scores" in text
    checks["predicate_target_geometry"] = all(s in text for s in ["box2d", "poly2d", "drivable_available", "left_corridor", "right_corridor", "predicate_reliability"])
    checks["label_trunk_uses_predicate_tokens"] = "predicate_cross_attn" in text and "predicate_tokens" in text and "label_self_attn" in text
    checks["predicate_reason_grammar_conditioned"] = all(s in text for s in ["positive_mask", "contradictory_mask", "predicate_reason_contradiction_score_by_label"])
    checks["supervisor_full_gate"] = all(s in Path("fate_oia/engine/supervise_acpr_oia_foreground.py").read_text(encoding="utf-8") for s in ["audit_acpr_oia_implementation", "max_train_samples", "fallback_ladder", "GOAL_COMPLETED_ACPR_OIA_V1.json"])
    checks["dino_field_last_tokens"] = all(s in Path("fate_oia/models/acpr_dino_field.py").read_text(encoding="utf-8") for s in ["patch_tokens_last", "cls_token_last", "original_tokens"])
    checks["pair_memory_enqueue"] = all(s in Path("fate_oia/models/acpr_pair_memory.py").read_text(encoding="utf-8") for s in ["def enqueue", "def mine", "pair_neg_memory_indices", "reason_logits_detached", "reason_embeddings_detached"])
    checks["combo_cardinality_outputs"] = all(s in Path("fate_oia/models/acpr_action_combo_aux.py").read_text(encoding="utf-8") for s in ["cardinality_logits", "combo_stats"])
    checks["calibration_split_outputs"] = all(s in Path("fate_oia/models/acpr_calibration.py").read_text(encoding="utf-8") for s in ["action_logits_calibrated", "reason_logits_calibrated", "bias_action", "temperature_reason"])
    checks["model_contract_outputs"] = all(s in Path("fate_oia/models/acpr_oia_model.py").read_text(encoding="utf-8") for s in ["direct_plus_predicate", "action_logits_raw", "reason_logits_raw", "cardinality_logits", "reason_embeddings_for_pair"])
    checks["train_artifact_schema"] = all(s in Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8") for s in [
        "implementation_fingerprint.json",
        "logits_action_raw_test.pt",
        "logits_reason_raw_test.pt",
        "predicate_logits_test.pt",
        "pair_cases_test.jsonl",
        "pair_mining_stats.jsonl",
        "pair_margin_per_reason.json",
        "matched_counterfactual_cases.jsonl",
        "checkpoint_best_test_tail_mf1.pth",
    ])
    checks["eval_diagnostics_schema"] = all(s in Path("fate_oia/engine/eval_acpr_oia.py").read_text(encoding="utf-8") for s in ["action_composition", "tail_reason", "predicate_group", "pair_margin"])
    checks["visual_export_counterfactual_schema"] = all(s in Path("fate_oia/engine/export_acpr_visuals.py").read_text(encoding="utf-8") for s in ["matched_negative", "predicate_delta", "reason_margin", "report.html"])
    forbidden = ["frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits", "tail_residual_adapter", "Start-Process", "Start-Job", "nohup"]
    checks["forbidden_patterns_absent"] = not any(pat in text for pat in forbidden)
    from fate_oia.models.acpr_oia_model import ACPROIAModel
    model = ACPROIAModel(use_mock_dino=True)
    forward_out = model(torch.randn(2, 3, 360, 640))
    checks["full_model_forward_contract"] = all(k in forward_out for k in [
        "patch_tokens_last",
        "cls_token_last",
        "action_logits_raw",
        "reason_logits_raw",
        "action_logits_calibrated",
        "reason_logits_calibrated",
        "contradiction_score",
        "required_support_score",
        "cardinality_logits",
        "reason_embeddings_for_pair",
        "action_token_norm_mean",
    ]) and forward_out["action_logits_raw"].shape == (2, 4) and forward_out["reason_logits_raw"].shape == (2, 21)
    from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder
    from fate_oia.losses.acpr_losses import matched_pair_logit_loss
    builder = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    synthetic = [{
        "labels": [
            {"category": "car", "box2d": {"x1": 580, "y1": 400, "x2": 710, "y2": 680}},
            {"category": "lane", "poly2d": [[{"x": 240, "y": 500}, {"x": 220, "y": 710}]]},
        ],
        "drivable_available": True,
    }]
    pred = builder.build_from_records(synthetic)
    checks["predicate_synthetic_nonzero"] = bool(pred["predicate_targets"].sum().item() > 0 and pred["predicate_coverage"]["object_box_count"] > 0 and pred["predicate_coverage"]["lane_poly_count"] > 0 and pred["predicate_coverage"]["drivable_count"] > 0)
    logits = torch.zeros(3, 21, requires_grad=True)
    pairs = {"pair_pos_indices": torch.tensor([0]), "pair_neg_indices": torch.tensor([1]), "pair_reason_ids": torch.tensor([5]), "pair_weights": torch.tensor([1.0]), "pair_active_mask": torch.tensor([True]), "pair_neg_is_memory": torch.tensor([False])}
    loss = matched_pair_logit_loss(logits, pairs)
    loss.backward()
    grad = logits.grad
    checks["pair_loss_only_reason_r"] = bool(grad[:, 5].abs().sum() > 0 and grad[:, [i for i in range(21) if i != 5]].abs().sum() == 0)
    mem_logits = torch.zeros(2, 21, requires_grad=True)
    mem_pairs = {
        "pair_pos_indices": torch.tensor([0]),
        "pair_neg_indices": torch.tensor([-1]),
        "pair_neg_memory_indices": torch.tensor([7]),
        "pair_reason_ids": torch.tensor([3]),
        "pair_weights": torch.tensor([1.0]),
        "pair_active_mask": torch.tensor([True]),
        "pair_neg_is_memory": torch.tensor([True]),
        "pair_neg_logits_detached": torch.tensor([1.0]),
    }
    mem_loss = matched_pair_logit_loss(mem_logits, mem_pairs)
    checks["memory_pair_loss_no_batch_index"] = bool(torch.isfinite(mem_loss) and float(mem_loss.detach()) > 0)
    if calalign_enabled:
        from fate_oia.losses.acpr_threshold_losses import calalign_loss_bundle
        from fate_oia.models.acpr_threshold_head import ACPRThresholdHead
        from fate_oia.utils.acpr_threshold_search import search_best_thresholds_for_f1
        from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices

        head = ACPRThresholdHead()
        action_base = torch.tensor([[0.2, -0.1, 1.0, -1.0], [1.2, -0.5, -0.2, 0.3]], requires_grad=True)
        reason_base = torch.randn(2, 21, requires_grad=True)
        th_out = head(action_base, reason_base)
        base = torch.cat([action_base, reason_base], dim=-1)
        checks["calalign_threshold_head_contract"] = (
            th_out["logits_base"].shape == (2, 25)
            and th_out["logits_deploy"].shape == (2, 25)
            and th_out["threshold_logit"].shape == (25,)
            and torch.allclose(th_out["logits_deploy"], base - th_out["threshold_logit"].view(1, -1), atol=1e-6)
            and bool((th_out["action_threshold_prob"] >= 0.10 - 1e-6).all())
            and bool((th_out["reason_threshold_prob"] <= 0.85 + 1e-6).all())
        )
        synth_logits = torch.tensor([[-2.0], [-0.5], [0.2], [2.0]])
        synth_targets = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        search = search_best_thresholds_for_f1(synth_logits, synth_targets, grid=torch.tensor([0.20, 0.50, 0.80]))
        checks["calalign_threshold_search_contract"] = bool(search["threshold_prob"].shape == (1,) and search["best_f1"][0] >= 0.99)
        detached_action = action_base.detach()
        detached_reason = reason_base.detach()
        detached_action.requires_grad_(False)
        detached_reason.requires_grad_(False)
        losses = calalign_loss_bundle(
            head(detached_action, detached_reason)["action_logits_deploy"],
            head(detached_action, detached_reason)["reason_logits_deploy"],
            torch.zeros(2, 4),
            torch.zeros(2, 21),
            head.compose_theta(),
            head.theta_teacher,
            head.train_prior_theta,
            head.teacher_pred_rate,
        )
        losses["total"].backward()
        checks["calalign_loss_updates_threshold_only"] = bool(head.theta_delta.grad is not None and head.theta_delta.grad.abs().sum() > 0 and action_base.grad is None and reason_base.grad is None)
        cal_model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True)
        cal_out = cal_model(torch.randn(2, 3, 360, 640))
        checks["calalign_model_forward_contract"] = all(k in cal_out for k in [
            "action_logits_base",
            "reason_logits_base",
            "action_logits_deploy",
            "reason_logits_deploy",
            "threshold_logit",
            "threshold_prob",
        ]) and torch.allclose(cal_out["action_logits_final_raw"], cal_out["action_logits_deploy"]) and torch.allclose(cal_out["branch_logits"]["base_fixed"], cal_out["logits_base_fixed"])
        old_model = ACPROIAModel(use_mock_dino=True, threshold_enabled=False)
        old_out = old_model(torch.randn(1, 3, 360, 640))
        checks["calalign_old_config_threshold_disabled"] = torch.allclose(old_out["action_logits_final_raw"], old_out["action_logits_base"]) and torch.allclose(old_out["reason_logits_final_raw"], old_out["reason_logits_base"])
        train_text = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
        checks["calalign_train_calib_no_test_leakage"] = all(s in train_text for s in [
            "collect_threshold_teacher",
            "train_calib_loader",
            "update_threshold_teacher_from_train_calib",
            "test_oracle",
        ]) and "copy_test_threshold" not in train_text
        cfg_eval = cfg.get("eval", {})
        checks["calalign_config_protocol"] = (
            cfg.get("threshold", {}).get("enabled") is True
            and cfg_eval.get("primary_raw_branch") in {"deploy_fixed", "actalign_utility_deploy_fixed", "candidate_probe_guarded_fallback"}
            and cfg_eval.get("also_eval_base_fixed") is True
            and cfg.get("feature_cache_enabled") is False
            and cfg.get("token_compression") == "none"
        )
        class Tiny:
            samples = [
                type("Sample", (), {"file_name": "b.jpg"})(),
                type("Sample", (), {"file_name": "a.jpg"})(),
                type("Sample", (), {"file_name": "c.jpg"})(),
                type("Sample", (), {"file_name": "d.jpg"})(),
            ]
            def __len__(self): return len(self.samples)
        main_idx, calib_idx = make_train_calib_indices(Tiny(), calib_fraction=0.5, seed=11)
        checks["calalign_train_calib_split_contract"] = bool(main_idx and calib_idx and not (set(main_idx) & set(calib_idx)) and (main_idx, calib_idx) == make_train_calib_indices(Tiny(), calib_fraction=0.5, seed=11))
        checks["calalign_script_foreground"] = "Start-Process" not in Path("scripts/FATE_OIA_acpr_calalign_v1_2_foreground.ps1").read_text(encoding="utf-8")
    if actalign_enabled:
        act_cfg = cfg.get("actalign", {})
        model_text = Path("fate_oia/models/acpr_oia_model.py").read_text(encoding="utf-8")
        train_text = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
        utility_text = Path("fate_oia/models/acpr_action_utility.py").read_text(encoding="utf-8")
        pred_text = Path("fate_oia/models/acpr_action_predicate_delta.py").read_text(encoding="utf-8")
        gate_text = Path("fate_oia/utils/acpr_action_pareto_gate.py").read_text(encoding="utf-8")
        guard_text = Path("fate_oia/utils/acpr_action_gradient_guard.py").read_text(encoding="utf-8")
        checks["actalign_forbidden_arch_absent"] = not any(x in model_text + train_text for x in ["MoE", "specialist", "selector_primary", "graph_delta_to_logits=True", "action_set_probs @"])
        checks["actalign_config_protocol"] = (
            cfg.get("model", {}).get("action_set_affects_final_action") is False
            and cfg.get("model", {}).get("graph_delta_to_logits") is False
            and cfg.get("runtime", {}).get("no_feature_cache") is True
            and cfg.get("runtime", {}).get("require_no_token_compression") is True
            and cfg.get("runtime", {}).get("test_only") is True
            and float(act_cfg.get("max_pred_delta", 9.0)) <= 0.05
            and float(act_cfg.get("max_r2a_delta", 9.0)) <= 0.20
            and float(act_cfg.get("initial_r2a_gate", 1.0)) == 0.0
            and float(act_cfg.get("initial_pred_gate", 1.0)) == 0.0
        )
        checks["actalign_utility_formula"] = all(t in utility_text for t in ["action_reason_logits - action_visual_logits", "action_logits_fallback + r2a_gate * r2a_delta + pred_gate * pred_delta", 'register_buffer("r2a_gate"', 'register_buffer("pred_gate"'])
        checks["actalign_predicate_delta_contract"] = all(t in pred_text for t in ["action_nodes", "predicate_probs", "detach_inputs", "nn.init.zeros_", "torch.tanh(raw) * self.max_delta"])
        torch.manual_seed(123)
        disabled = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=False)
        torch.manual_seed(123)
        enabled = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=True, actalign_kwargs={"initial_r2a_gate": 0.0, "initial_pred_gate": 0.0})
        x = torch.randn(2, 3, 360, 640)
        out_d = disabled(x)
        out_e = enabled(x)
        checks["actalign_zero_gate_exact_action"] = bool(torch.allclose(out_d["action_logits_final_raw"], out_e["action_logits_final_raw"], atol=1e-6))
        checks["actalign_reason_unchanged_by_utility"] = bool(torch.allclose(out_d["reason_logits_final_raw"], out_e["reason_logits_final_raw"], atol=1e-6))
        enabled.action_utility.set_gates(r2a_gate=torch.tensor([1.0, 0.0, 0.0, 0.0]), pred_gate=torch.zeros(4))
        out_one = enabled(x)
        diff = (out_one["action_logits_final_raw"] - out_e["action_logits_final_raw"]).abs()
        checks["actalign_single_gate_only_selected_action"] = bool(diff[:, 0].max() > 0 and diff[:, 1:].max() < 1e-6)
        checks["actalign_delta_bounds"] = bool(out_one["action_r2a_delta"].abs().max() <= 0.20001 and out_one["action_predicate_delta"].abs().max() <= 0.05001)
        from fate_oia.utils.acpr_action_pareto_gate import ActionParetoGate
        gate = ActionParetoGate(action_dim=4, gate_ema=1.0, min_support=1)
        labels = torch.zeros(4, 4); labels[:2, 0] = 1
        base_logits = torch.zeros(4, 4)
        r2a_logits = base_logits.clone(); r2a_logits[:2, 0] = 4; r2a_logits[2:, 0] = -4
        gate_stats = gate.update(base_logits, r2a_logits, base_logits, labels)
        checks["actalign_gate_train_calib_only"] = gate_stats.get("source") == "train_calib_only" and gate.r2a_gate[0].item() == 1.0
        checks["actalign_artifact_writers"] = all(t in train_text for t in ["action_utility_metrics.jsonl", "action_utility_gates.jsonl", "gradient_guard_stats.jsonl", "cooldown_stats.jsonl", "ema_swa_metrics.jsonl", "checkpoint_best_test_action_primary.pth"])
        checks["actalign_gate_uses_f1_not_bce"] = all(t in gate_text for t in ["_binary_f1_per_label", "F1_base_per_action", "F1_r2a_candidate_per_action", "delta_r2a_per_action"]) and "binary_cross_entropy" not in gate_text
        checks["actalign_gradient_guard_projects_in_trainer"] = all(t in guard_text for t in ["capture_action_grads", "project_model_grads", "param.grad.copy_"]) and all(t in train_text for t in ["grad_guard.capture_action_grads", "grad_guard.project_model_grads"])
        checks["actalign_ema_swa_real_eval"] = all(t in train_text for t in ["evaluate_shadow_model", "average_parameters", "averaged_parameters", "swa_helper.consider", "metrics_ema", "metrics_swa", "checkpoint_best_test_action_primary_ema.pth"]) and '"deploy_fixed_ema": None' not in train_text
        checks["actalign_cooldown_changes_lr_and_weights"] = all(t in train_text for t in ["update_cooldown_state", "cooldown_multiplier", "weights[\"action_deploy\"]", "lr_multiplier_threshold", "action_visual_aux_bonus"])
        if candidate_probe:
            cand_text = Path("fate_oia/models/acpr_action_candidates.py").read_text(encoding="utf-8")
            cand_loss_text = Path("fate_oia/losses/acpr_candidate_losses.py").read_text(encoding="utf-8")
            cand_gate_text = Path("fate_oia/utils/acpr_candidate_gate.py").read_text(encoding="utf-8")
            checks["candidate_files_contract"] = all(t in cand_text for t in ["ACPRActionCandidates", "blend_gamma_raw", "selected_candidate_id", "set_selected_candidates", "utility_final"])
            checks["candidate_loss_contract"] = all(t in cand_loss_text for t in ["all_candidate_probe_loss", "action_candidate_nonregression_loss", "candidate_action_asl_loss"])
            checks["candidate_gate_train_calib_contract"] = all(t in cand_gate_text for t in ["update_from_train_calib", "selected_candidate_", "pred_rate_explosion", "all_high_increase", "delta_f1_"])
            candidate_model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, actalign_enabled=True, actalign_kwargs={"mode": "candidate_probe"})
            candidate_out = candidate_model(torch.randn(2, 3, 360, 640))
            required_candidates = {"fallback", "visual", "reason", "blend", "predicate", "blend_predicate"}
            checks["candidate_forward_contract"] = (
                "action_candidate_logits" in candidate_out
                and required_candidates.issubset(set(candidate_out["action_candidate_logits"].keys()))
                and torch.allclose(candidate_out["action_logits_utility"], candidate_out["action_logits_fallback"], atol=1e-6)
            )
            candidate_loss = candidate_out["action_candidate_logits"]["blend"].sum() + candidate_out["action_candidate_logits"]["predicate"].sum()
            candidate_loss.backward()
            blend_grad = candidate_model.action_candidates.blend_gamma_raw.grad
            pred_grad = candidate_model.action_predicate_delta.mlp[-1].weight.grad
            checks["candidate_gradients_active"] = bool(blend_grad is not None and blend_grad.abs().sum() > 0 and pred_grad is not None and pred_grad.abs().sum() > 0)
            checks["candidate_stage_protocol"] = (
                cfg.get("stageA", {}).get("train_candidate_heads_only") is True
                and cfg.get("stageB", {}).get("enabled") is False
                and cfg.get("stageB", {}).get("require_stageA_pass") is True
                and all(t in train_text for t in ["STAGE_A_CANDIDATE_PROBE_PASS.json", "STAGE_A_CANDIDATE_PROBE_FAIL.json", "collect_action_candidates_train_calib"])
                and "candidate_probe" in Path("fate_oia/engine/fit_acpr_action_candidates.py").read_text(encoding="utf-8")
            )
    pass_all = not missing and all(checks.values())
    pass_name = "REVIEW_PASS_ACPR_ACTALIGN_V1_3_1.txt" if candidate_probe else ("REVIEW_PASS_ACPR_ACTALIGN_V1_3.txt" if actalign_enabled else "REVIEW_PASS_ACPR_OIA_V1.txt")
    pass_label = "REVIEW_PASS_ACPR_ACTALIGN_V1_3_1" if candidate_probe else ("REVIEW_PASS_ACPR_ACTALIGN_V1_3" if actalign_enabled else "REVIEW_PASS_ACPR_OIA_V1")
    result = {
        "pass": pass_all,
        "git_head": _git_head(),
        "checked_files": required_files,
        "forbidden_pattern_results": {"forbidden_patterns_absent": checks.get("forbidden_patterns_absent", False)},
        "functional_checks": checks,
        "smoke_result": {"required_before_full_train": True, "checked_by_supervisor": True},
        "review_pass_path": str(out_dir / pass_name),
        "missing_items": missing + [k for k, v in checks.items() if not v],
        "warnings": [],
    }
    (out_dir / "implementation_audit_ACPR_OIA_V1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if pass_all and args.write_review_pass:
        (out_dir / pass_name).write_text(pass_label + "\n" + result["git_head"] + "\n", encoding="utf-8")
    if not pass_all:
        raise SystemExit(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

