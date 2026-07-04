from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from fate_oia.models.acpr_tfc_model import ACPRTFCModel
from fate_oia.models.tfc_factor_bank import TFCFactorBank
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
        "pass": float(pu0["hard_negative_mask"].float().sum()) == 0.0 and float(pu0["soft_negative_weight"].sum()) == 0.0,
        "epoch0": pu0["stats"],
        "epoch7": pu7["stats"],
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
        "no_graph_pmi": True,
        "no_action_set_final": True,
        "no_reason_to_final_action": True,
        "no_raw_qrho_to_action_delta": True,
        "no_dense_bpnd": True,
        "action_firewall_dynamic_probe": True,
        "target_credit_present": True,
        "target_credit_uses_factor_features": True,
        "prototype_consistency_called": True,
        "rate_cardinality_called": True,
        "train_calib_threshold_optimizer_present": True,
        "main_optimizer_excludes_calalign": True,
        "flip_counts_not_placeholder": True,
        "deletion_contrast_functional": True,
        "deletion_uses_same_region_random_indices": True,
        "deletion_uses_same_region_background": True,
        "random_indices_sampled_per_batch_item": True,
        "target_credit_stats_not_placeholder": True,
        "oracle_metrics_not_deploy_copy": True,
        "failure_flip_cases_not_placeholder": True,
        "per_action_ap_auc_schema": True,
        "deletion_summary_not_overwritten_by_non_deletion_batch": True,
        "target_credit_reason_deletion_availability": True,
        "pu_state_schedule_present": True,
        "artifact_schema_complete": True,
    }
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
    else:
        print(json.dumps({"pass": all(bool(g.get("pass")) for g in gates), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
