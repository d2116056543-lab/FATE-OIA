from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

import torch
import yaml

# Threshold deploy semantics: deploy = base - theta
REQUIRED = [
    "fate_oia/models/acpr_fusionlite_gate.py",
    "fate_oia/utils/acpr_action_semantic_maps.py",
    "fate_oia/losses/acpr_fusionlite_losses.py",
    "fate_oia/utils/acpr_model_ema.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_oia_model.py",
    "fate_oia/models/acpr_threshold_head.py",
    "fate_oia/engine/train_acpr_oia.py",
    "fate_oia/engine/supervise_acpr_oia_foreground.py",
    "configs/fate_oia_train_360x640_acpr_fusionlite_v1_4.yaml",
    "scripts/FATE_OIA_acpr_fusionlite_v1_4_foreground.ps1",
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    checks: dict[str, bool] = {}
    missing: list[str] = []
    for rel in REQUIRED:
        p = Path(rel)
        if not p.exists():
            missing.append(rel)
            continue
        if p.suffix == ".py":
            ast.parse(p.read_text(encoding="utf-8"))
    text = "\n".join(Path(rel).read_text(encoding="utf-8") for rel in REQUIRED if Path(rel).exists())
    forbidden = [
        "frozen_run_c",
        "FrozenRunC",
        "run_c_logits",
        "cached_logits",
        "tail_residual_adapter",
        "Start-Process",
        "Start-Job",
        "nohup",
        "hidden cmd",
        "scheduled task",
        "daemon",
        "action_set_probs @ subset_membership",
    ]
    checks["forbidden_patterns_absent"] = not any(p in text for p in forbidden)
    checks["config_no_cache_no_compression_test"] = (
        cfg.get("feature_cache_enabled") is False
        and cfg.get("token_compression") == "none"
        and cfg.get("eval_splits") == "test"
        and cfg.get("best_selection_split") == "test"
    )
    checks["fusionlite_config_enabled"] = cfg.get("model", {}).get("use_fusionlite") is True and cfg.get("model", {}).get("threshold_enabled") is True and cfg.get("fusionlite", {}).get("zero_init_delta") is True
    checks["loss_section_present"] = all(k in cfg.get("loss", {}) for k in ["action_deploy_weight", "fusionlite_delta_l2_weight", "r2a_forbidden_prior_weight"])
    checks["audit_section_present"] = cfg.get("audit", {}).get("export_online_and_ema_metrics") is True and cfg.get("audit", {}).get("export_gate_delta_table") is True
    checks["files_have_core_flow"] = all(
        s in text
        for s in ["ACPRFusionLiteGate", "load_action_semantic_maps", "action_logits_direct_legacy", "fusionlite_delta_gate", "deploy = base - theta"]
    )
    from fate_oia.utils.acpr_action_semantic_maps import load_action_semantic_maps

    maps = load_action_semantic_maps("configs/acpr_reason_predicate_grammar.yaml", "configs/acpr_scene_predicates.yaml")
    checks["semantic_masks_shape"] = maps.action_reason_mask.shape == (4, 21) and maps.action_predicate_mask.shape[0] == 4 and maps.forbidden_r2a_mask.shape == (4, 21)
    checks["semantic_uses_spatial_region"] = "spatial_region" in Path("fate_oia/utils/acpr_action_semantic_maps.py").read_text(encoding="utf-8") and "region_to_pred_ids" in Path("fate_oia/utils/acpr_action_semantic_maps.py").read_text(encoding="utf-8")
    from fate_oia.models.acpr_fusionlite_gate import ACPRFusionLiteGate

    gate = ACPRFusionLiteGate(dim=8, num_predicates=max(int(maps.action_predicate_mask.shape[1]), 1))
    final = gate.delta_mlp[-1]
    checks["fusionlite_final_zero_init"] = bool(torch.allclose(final.weight, torch.zeros_like(final.weight)) and torch.allclose(final.bias, torch.zeros_like(final.bias)))
    from fate_oia.losses.acpr_fusionlite_losses import fusionlite_delta_l2, r2a_forbidden_prior_loss

    checks["fusionlite_losses_finite"] = bool(torch.isfinite(fusionlite_delta_l2(torch.zeros(2, 4))) and torch.isfinite(r2a_forbidden_prior_loss(torch.ones(4, 21), maps.forbidden_r2a_mask)))
    from fate_oia.models.acpr_oia_model import ACPROIAModel

    torch.manual_seed(11)
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, use_fusionlite=True)
    x = torch.randn(2, 3, 360, 640)
    out = model(x)
    checks["model_forward_shapes"] = out["action_logits_base"].shape == (2, 4) and out["reason_logits_base"].shape == (2, 21) and out["action_fusion_gate"].shape == (2, 4)
    checks["dino_frozen"] = all(not p.requires_grad for p in model.dino.parameters())
    checks["zero_init_compat"] = bool(
        torch.allclose(out["action_logits_base"], out["action_logits_direct_legacy"], atol=1e-6)
        and torch.allclose(out["action_logits_fusionlite"], out["action_logits_direct_legacy"], atol=1e-6)
    )
    checks["deploy_semantics"] = bool(torch.allclose(out["logits_deploy"], out["logits_base_fixed"] - out["threshold_logit"].view(1, -1), atol=1e-6))
    model2 = ACPROIAModel(use_mock_dino=True, threshold_enabled=False, use_fusionlite=False)
    out2 = model2(x)
    checks["old_config_still_runs"] = out2["action_logits_base"].shape == (2, 4) and "fusionlite_delta_gate" in out2
    train_text = Path("fate_oia/engine/train_acpr_oia.py").read_text(encoding="utf-8")
    checks["ema_actual_created"] = "ema_helper = ModelEMA" in train_text and "apply_to(model)" in train_text and "restore(model, backup)" in train_text
    checks["cooldown_actual"] = "train_calib_action_primary_scores" in train_text and "action_primary_cooldown_should_trigger" in train_text and "fusionlite_cooldown_events.jsonl" in train_text
    checks["fusionlite_artifacts_complete"] = all(s in train_text for s in ["fusionlite_metrics.jsonl", "fusionlite_gate_stats.jsonl", "fusionlite_per_action_table.json", "deploy_fixed_ema", "old_gate_mean", "new_gate_mean", "visual_F1", "reason_F1", "legacy_F1", "fusionlite_F1"])
    checks["reason_unchanged_by_fusionlite_contract"] = "reason_logits_base = trunk[\"reason_logits_visual\"] + reason_delta" in Path("fate_oia/models/acpr_oia_model.py").read_text(encoding="utf-8")
    pass_all = not missing and all(checks.values())
    result = {
        "pass": pass_all,
        "git_head": _git_head(),
        "checked_files": REQUIRED,
        "forbidden_pattern_results": {"forbidden_patterns_absent": checks.get("forbidden_patterns_absent", False)},
        "functional_checks": checks,
        "smoke_result": {"dynamic_forward": checks.get("model_forward_shapes", False)},
        "review_pass_path": str(out_dir / "REVIEW_PASS_ACPR_FUSIONLITE_V1_4.txt"),
        "missing_items": missing + [k for k, v in checks.items() if not v],
        "warnings": maps.warnings,
    }
    (out_dir / "implementation_audit_ACPR_FUSIONLITE_V1_4.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if pass_all and args.write_review_pass:
        (out_dir / "REVIEW_PASS_ACPR_FUSIONLITE_V1_4.txt").write_text("REVIEW_PASS_ACPR_FUSIONLITE_V1_4\n" + result["git_head"] + "\n", encoding="utf-8")
    if not pass_all:
        raise SystemExit(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
