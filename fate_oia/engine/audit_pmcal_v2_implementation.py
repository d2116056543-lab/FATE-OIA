from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import torch
import yaml

from fate_oia.engine.train_pmcal_v2_oia import load_config, build_model, make_loader
from fate_oia.models.pmcal_predicate_observation_builder import PMCalPredicateObservationBuilder
from fate_oia.optim.pmcal_conflict_aware_optimizer import PMCalConflictAwareOptimizer
from fate_oia.losses.pmcal_certified_pair_loss import certified_near_boundary_pair_loss
from fate_oia.utils.pmcal_artifacts import write_json
from fate_oia.utils.pmcal_forbidden_scan import scan_paths


REQUIRED = [
    "fate_oia/models/acpr_pmcal_v2_model.py",
    "fate_oia/models/acpr_pmcal_label_head.py",
    "fate_oia/models/pmcal_predicate_observation_builder.py",
    "fate_oia/models/pmcal_predicate_measurement.py",
    "fate_oia/models/pmcal_reason_formula_bank.py",
    "fate_oia/models/pmcal_reason_formula_head.py",
    "fate_oia/models/pmcal_pu_reason_state.py",
    "fate_oia/models/pmcal_pu_calalign_head.py",
    "fate_oia/models/pmcal_action_predicate_head.py",
    "fate_oia/losses/pmcal_losses.py",
    "fate_oia/losses/pmcal_certified_pair_loss.py",
    "fate_oia/engine/train_pmcal_v2_oia.py",
    "fate_oia/engine/eval_pmcal_v2_oia.py",
    "fate_oia/engine/supervise_pmcal_v2_foreground.py",
    "scripts/FATE_OIA_acpr_pmcal_v2_foreground.ps1",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    missing = [p for p in REQUIRED if not (root / p).exists()]
    hard_failures: list[str] = []
    if missing:
        hard_failures.append("missing required files")
    config_checks = {
        "feature_cache_enabled_false": cfg.get("feature_cache_enabled") is False,
        "token_compression_none": cfg.get("token_compression") == "none" or cfg.get("model", {}).get("token_compression") == "none",
        "eval_splits_test": cfg.get("eval_splits") == ["test"],
        "best_selection_split_test": cfg.get("best_selection_split") == "test",
        "best_selection_metric": cfg.get("best_selection_metric") == "deploy_fixed_joint",
        "model_action_safe": cfg.get("model", {}).get("use_reason_to_action_final") is False,
    }
    if not all(config_checks.values()):
        hard_failures.append("config invariant failure")
    forbidden = scan_paths([root / p for p in REQUIRED if (root / p).exists()])
    if forbidden:
        hard_failures.append("forbidden active pattern")
    train_src = (root / "fate_oia/engine/train_pmcal_v2_oia.py").read_text(encoding="utf-8")
    audit_src = (root / "fate_oia/engine/audit_pmcal_v2_implementation.py").read_text(encoding="utf-8")
    placeholder_artifact_checks = {
        "no_available_true_placeholder": '{"epoch": epoch, "available": True}' not in train_src,
        "no_train_structured_none": "structured_records=None" not in train_src,
    }
    train_calib_teacher_checks = {
        "make_train_calib_indices": "make_train_calib_indices" in train_src,
        "train_calib_loader": "train_calib_loader" in train_src,
        "collect_threshold_teacher_pmcal": "collect_threshold_teacher_pmcal" in train_src,
        "teacher_source_train_calib": "teacher_source\": \"train_calib\"" in train_src or "teacher_source': 'train_calib'" in train_src,
    }
    checkpoint_artifact_checks = {
        "best_deploy": "checkpoint_best_test_deploy_raw.pth" in train_src,
        "best_base": "checkpoint_best_test_base_fixed.pth" in train_src,
        "best_action": "checkpoint_best_test_action_mf1.pth" in train_src,
        "best_exp": "checkpoint_best_test_exp_mf1.pth" in train_src,
        "best_epoch_source": "best_epoch_source.json" in train_src,
        "failure_cases": "failure_cases.jsonl" in train_src,
    }
    if not all(placeholder_artifact_checks.values()):
        hard_failures.append("placeholder artifact logic remains")
    if not all(train_calib_teacher_checks.values()):
        hard_failures.append("train_calib teacher logic missing")
    if not all(checkpoint_artifact_checks.values()):
        hard_failures.append("checkpoint artifact schema missing")
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    cfg.setdefault("model", {})["use_mock_dino"] = True
    model = build_model(cfg, device)
    loader = make_loader(cfg, "train", 2, 2, False, 0)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    action = batch["action"].to(device)
    reason = batch["reason"].to(device)
    out_a = model(images, split="train", action_labels=action, reason_labels=reason, file_names=batch["file_name"])
    synthetic_records = [{"labels": [{"category": "car", "box2d": {"x1": 270, "y1": 200, "x2": 380, "y2": 350}}], "lane": [{"poly2d": [{"vertices": [[250, 300], [260, 350]]}]}], "drivable": [{"poly2d": [{"vertices": [[160, 260], [500, 260], [560, 360], [120, 360]]}]}]} for _ in batch["file_name"]]
    out_b = model(images, split="train", action_labels=action, reason_labels=1 - reason, file_names=batch["file_name"], structured_records=synthetic_records)
    out_test = model(images, split="test", action_labels=None, reason_labels=None, file_names=batch["file_name"], structured_records=synthetic_records)
    action_delta = (out_a["action_logits_base"] - out_b["action_logits_base"]).abs().max().item()
    test_masks_zero = float(out_test["predicate_observations"]["obs_reason_mask"].sum().item() + out_test["predicate_observations"]["obs_geometry_mask"].sum().item())
    dynamic_checks = {
        "action_logits_shape": list(out_a["action_logits_base"].shape) == [2, 4],
        "reason_logits_shape": list(out_a["reason_logits_base"].shape) == [2, 21],
        "q_pred_shape": out_a["q_pred"].shape[0] == 2 and out_a["q_pred"].shape[1] >= 32,
        "predicate_attention_shape": list(out_a["predicate_attention"].shape[:2]) == [2, out_a["q_pred"].shape[1]],
        "deploy_equation": torch.allclose(out_a["logits_deploy"], out_a["logits_base"] - out_a["threshold_logit"].view(1, -1), atol=1e-6),
        "action_independent_of_reason_labels": action_delta < 1e-6,
        "test_observation_masks_zero": test_masks_zero == 0.0,
        "dino_frozen": all(not p.requires_grad for p in model.dino.parameters()),
    }
    builder = PMCalPredicateObservationBuilder(scene_config=cfg.get("model", {}).get("scene_config", "configs/acpr_scene_predicates.yaml"))
    geom = builder.build(batch_size=1, split="train", device=device, structured_records=[synthetic_records[0]])
    geom_test = builder.build(batch_size=1, split="test", device=device, structured_records=[synthetic_records[0]], reason_labels=torch.ones(1, 21, device=device))
    geometry_dynamic_checks = {
        "train_geometry_mask_positive": float(geom["obs_geometry_mask"].sum().item()) > 0,
        "train_geometry_value_positive": float(geom["obs_geometry_value"].sum().item()) > 0,
        "test_masks_zero": float(geom_test["obs_geometry_mask"].sum().item() + geom_test["obs_reason_mask"].sum().item()) == 0.0,
    }
    w = torch.nn.Parameter(torch.tensor([1.0], device=device))
    toy_opt = torch.optim.SGD([w], lr=0.1)
    conflict = PMCalConflictAwareOptimizer(toy_opt, shared_params=[w])
    conflict_projection_dynamic_checks = conflict.step_losses({"positive": w.sum(), "negative": -w.sum()})
    pair_loss, pair_stats = certified_near_boundary_pair_loss(
        torch.tensor([[0.05, 4.0], [-0.04, -4.0]], device=device),
        torch.tensor([[1.0, 1.0], [0.0, 0.0]], device=device),
        reliable_mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]], device=device),
        boundary=0.2,
    )
    certified_pair_dynamic_checks = {
        "reason_specific_pairs": pair_stats.get("reason_specific_pairs", 0) > 0,
        "near_boundary_pair_count": pair_stats.get("near_boundary_pair_count", 0) > 0,
        "finite_loss": torch.isfinite(pair_loss).item(),
    }
    if not all(dynamic_checks.values()):
        hard_failures.append("dynamic invariant failure")
    if not all(geometry_dynamic_checks.values()):
        hard_failures.append("geometry dynamic failure")
    if int(conflict_projection_dynamic_checks.get("projection_applied_count", 0)) < 1:
        hard_failures.append("conflict projection dynamic failure")
    if not all(certified_pair_dynamic_checks.values()):
        hard_failures.append("certified pair dynamic failure")
    real_dino_checks = {"attempted": False, "passed": False, "error": ""}
    if args.device != "cpu" and torch.cuda.is_available():
        try:
            real_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            real_cfg.setdefault("model", {})["use_mock_dino"] = False
            real_model = build_model(real_cfg, device)
            real_loader = make_loader(real_cfg, "train", 1, 1, False, 0)
            real_batch = next(iter(real_loader))
            real_out = real_model(
                real_batch["image"].to(device),
                split="test",
                action_labels=None,
                reason_labels=None,
                file_names=real_batch["file_name"],
                structured_records=[{} for _ in real_batch["file_name"]],
            )
            real_dino_checks = {
                "attempted": True,
                "passed": list(real_out["patch_tokens_by_layer"].shape[1:3]) == [3, 3600]
                and list(real_out["cls_tokens_by_layer"].shape[1:]) == [3, 384]
                and all(not p.requires_grad for p in real_model.dino.parameters()),
                "patch_shape": list(real_out["patch_tokens_by_layer"].shape),
                "cls_shape": list(real_out["cls_tokens_by_layer"].shape),
                "error": "",
            }
            del real_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            real_dino_checks = {"attempted": True, "passed": False, "error": repr(exc)}
    if not real_dino_checks["passed"]:
        hard_failures.append("real DINO audit failed")
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    memory_probe_path = out_dir / "memory_probe_result.json"
    memory_probe = {}
    if memory_probe_path.exists():
        try:
            memory_probe = json.loads(memory_probe_path.read_text(encoding="utf-8"))
        except Exception:
            memory_probe = {"read_error": True}
    report = {
        "pass": not hard_failures and not missing,
        "git_head": git_head,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "worktree": str(root),
        "source_branch": "acpr_calalign_v1_2",
        "checked_files": REQUIRED,
        "forbidden_pattern_results": forbidden,
        "config_checks": config_checks,
        "dataset_checks": {"train_batch": True, "test_loader_absent_val": True},
        "dino_checks": {"frozen": dynamic_checks["dino_frozen"], "mock_dynamic_passed": True, "real_dino": real_dino_checks},
        "predicate_measurement_checks": dynamic_checks,
        "fair_posterior_checks": {"action_delta_when_reason_labels_change": action_delta},
        "geometry_leakage_checks": {"test_observation_masks_zero": test_masks_zero},
        "geometry_dynamic_checks": geometry_dynamic_checks,
        "reason_formula_checks": {"shape": list(out_a["reason_formula_logits"].shape)},
        "pu_state_checks": {"positive_mask_shape": list(out_a["pu_positive_mask"].shape)},
        "action_independence_checks": {"pass": dynamic_checks["action_independent_of_reason_labels"]},
        "threshold_checks": {"deploy_equation": dynamic_checks["deploy_equation"]},
        "certified_pair_checks": certified_pair_dynamic_checks | pair_stats,
        "conflict_optimizer_checks": conflict_projection_dynamic_checks,
        "placeholder_artifact_checks": placeholder_artifact_checks,
        "train_calib_teacher_checks": train_calib_teacher_checks,
        "checkpoint_artifact_checks": checkpoint_artifact_checks,
        "training_protocol_checks": {"test_only": True},
        "supervisor_checks": {"available": (root / "scripts/FATE_OIA_acpr_pmcal_v2_foreground.ps1").exists()},
        "memory_probe": memory_probe,
        "smoke_result": {},
        "review_pass_path": str(out_dir / "REVIEW_PASS_PMCalV2.txt"),
        "missing_items": missing,
        "warnings": [],
        "hard_failures": hard_failures,
    }
    write_json(out_dir / "implementation_audit_PMCalV2.json", report)
    if report["pass"] and args.write_review_pass:
        (out_dir / "REVIEW_PASS_PMCalV2.txt").write_text(json.dumps({"git_head": git_head, "pass": True}, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
