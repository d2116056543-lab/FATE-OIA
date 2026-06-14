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
    "fate_oia/models/acpr_predicate_targets.py",
    "fate_oia/models/acpr_pair_memory.py",
    "fate_oia/losses/acpr_losses.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_predicate_reason.py",
    "fate_oia/engine/supervise_acpr_oia_foreground.py",
    "fate_oia/engine/train_acpr_oia.py",
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
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    checks: dict[str, bool] = {}
    for rel in REQUIRED:
        p = Path(rel)
        if not p.exists():
            missing.append(rel); continue
        ast.parse(p.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    checks["test_only_no_cache_no_compression"] = cfg.get("eval_splits") == "test" and cfg.get("best_selection_split") == "test" and cfg.get("token_compression") == "none" and cfg.get("feature_cache_enabled") is False
    grammar = yaml.safe_load(Path("configs/acpr_reason_predicate_grammar.yaml").read_text(encoding="utf-8")) or {}
    display = yaml.safe_load(Path("configs/bdd_oia_reason_names_external.yaml").read_text(encoding="utf-8")) or {}
    checks["grammar_matches_external_names"] = all(str(grammar["reasons"][i]["name"]) == str(display["names"][i]) for i in range(21))
    checks["grammar_has_predicate_fields"] = all(all(k in grammar["reasons"][i] for k in ["positive_predicates", "contradictory_predicates", "compatible_actions", "hard_negative_reasons", "spatial_region"]) for i in range(21))
    text = "\n".join(Path(r).read_text(encoding="utf-8") for r in REQUIRED if Path(r).exists())
    checks["pair_mining_reason_specific"] = all(s in text for s in ["pair_reason_ids", "pair_pos_indices", "pair_neg_indices", "pair_contradiction", "global_embedding", "predicate_probs"])
    checks["pair_loss_reason_specific"] = "reason_logits[pos.long(), rid.long()]" in text and "margin - z_pos + z_neg" in text
    checks["pu_loss_uses_contradiction"] = "0.2 + (1.0 - neg_min_weight) * contradiction_scores" in text or "neg_min_weight + (1.0 - neg_min_weight) * contradiction_scores" in text
    checks["predicate_target_geometry"] = all(s in text for s in ["box2d", "poly2d", "drivable_available", "left_corridor", "right_corridor", "predicate_reliability"])
    checks["label_trunk_uses_predicate_tokens"] = "predicate_cross_attn" in text and "predicate_tokens" in text and "label_self_attn" in text
    checks["predicate_reason_grammar_conditioned"] = all(s in text for s in ["positive_mask", "contradictory_mask", "predicate_reason_contradiction_score_by_label"])
    checks["supervisor_full_gate"] = all(s in Path("fate_oia/engine/supervise_acpr_oia_foreground.py").read_text(encoding="utf-8") for s in ["audit_acpr_oia_implementation", "max_train_samples", "fallback_ladder", "GOAL_COMPLETED_ACPR_OIA_V1.json"])
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
    pairs = {"pair_pos_indices": torch.tensor([0]), "pair_neg_indices": torch.tensor([1]), "pair_reason_ids": torch.tensor([5]), "pair_weights": torch.tensor([1.0])}
    loss = matched_pair_logit_loss(logits, pairs)
    loss.backward()
    grad = logits.grad
    checks["pair_loss_only_reason_r"] = bool(grad[:, 5].abs().sum() > 0 and grad[:, [i for i in range(21) if i != 5]].abs().sum() == 0)
    pass_all = not missing and all(checks.values())
    result = {
        "pass": pass_all,
        "git_head": _git_head(),
        "checked_files": REQUIRED,
        "functional_checks": checks,
        "missing_items": missing + [k for k, v in checks.items() if not v],
        "warnings": [],
    }
    (out / "implementation_audit_ACPR_OIA_V1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if pass_all and args.write_review_pass:
        (out / "REVIEW_PASS_ACPR_OIA_V1.txt").write_text("REVIEW_PASS_ACPR_OIA_V1\n" + result["git_head"] + "\n", encoding="utf-8")
    if not pass_all:
        raise SystemExit(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
