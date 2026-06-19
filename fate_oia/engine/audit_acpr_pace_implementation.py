from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import yaml

from fate_oia.engine.train_acpr_oia import build_model, load_config


REQUIRED_FILES = [
    "fate_oia/models/acpr_predicate_action_coupling.py",
    "fate_oia/utils/acpr_pair_budget.py",
    "fate_oia/utils/acpr_pace_gradient_coordinator.py",
    "fate_oia/utils/acpr_pace_training_control.py",
    "fate_oia/utils/acpr_pace_artifacts.py",
    "fate_oia/utils/acpr_teacher_lock.py",
    "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
]

FORBIDDEN = [
    "Start-Process",
    "Start-Job",
    "nohup",
    "scheduled task",
    "frozen_run_c",
    "run_c_logits",
    "cached_logits",
    "feature_cache_enabled: true",
    "token_compression: keep_merge",
    "best_selection_split: val",
    "eval_splits: val",
    "graph_delta_to_logits: true",
    "MoE",
    "expert",
    "selector",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    root = Path.cwd()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    missing = [p for p in REQUIRED_FILES if not (root / p).exists()]
    text_targets = [
        root / "fate_oia/models/acpr_oia_model.py",
        root / "fate_oia/models/acpr_label_trunk.py",
        root / "fate_oia/losses/acpr_losses.py",
        root / "fate_oia/engine/train_acpr_oia.py",
        root / "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in text_targets if p.exists())
    forbidden_hits = {pat: (pat in combined) for pat in FORBIDDEN}
    checks = {
        "action_uses_predicate_conditioned_reason": "action_reason_logits_pace" in combined and "predicate_action_coupling" in combined,
        "exp_and_action_share_reason_delta": "reason_logits_base = trunk[\"reason_logits_visual\"] + reason_delta[\"predicate_reason_delta\"]" in combined,
        "pu_reason_losses": "pu_reason_soft_f1_loss" in combined and "pu_predicate_reason_alignment_loss" in combined,
        "pair_budget": "apply_pair_budget" in combined and "matched_pair_budget_ratio" in combined,
        "contribution_chain": "pace_action_reason_predicate_contrib_test.pt" in combined,
        "no_cache_no_compression": cfg.get("feature_cache_enabled") is False and cfg.get("token_compression") == "none",
        "test_only_best": cfg.get("eval_splits") == "test" and cfg.get("best_selection_split") == "test",
        "pace_config_enabled": bool(cfg.get("pace", {}).get("enabled", False)) and bool(cfg.get("model", {}).get("predicate_affects_action", False)),
        "official_batch": int(cfg.get("training", {}).get("batch_size", 0)) == 5 and int(cfg.get("training", {}).get("gradient_accumulation_steps", 0)) == 6,
        "weak_predicate_cleanup": "traffic_light_green" not in (root / "fate_oia/models/acpr_predicate_targets.py").read_text(encoding="utf-8", errors="ignore").split('if cat in {"traffic light"}:')[1].split('if cat in {"traffic sign"}:')[0],
    }
    smoke_result = {}
    try:
        dyn_cfg = dict(cfg)
        dyn_cfg.setdefault("model", {})
        dyn_cfg["model"] = dict(dyn_cfg.get("model", {}))
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        dyn_cfg["model"]["use_mock_dino"] = device.type != "cuda"
        model = build_model(dyn_cfg, device)
        model.eval()
        with torch.no_grad():
            result = model(torch.randn(2, 3, 360, 640, device=device), epoch=0)
        smoke_result = {
            "action_logits_base": list(result["action_logits_base"].shape),
            "reason_logits_base": list(result["reason_logits_base"].shape),
            "contrib_shape": list(result["predicate_reason_action_contrib_final"].shape),
        }
        checks["dynamic_forward"] = smoke_result["action_logits_base"] == [2, 4] and smoke_result["reason_logits_base"] == [2, 21]
    except Exception as exc:
        smoke_result = {"error": repr(exc)}
        checks["dynamic_forward"] = False
    pass_flag = not missing and all(checks.values()) and not any(forbidden_hits.values())
    payload = {
        "pass": pass_flag,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "checked_files": REQUIRED_FILES,
        "missing_items": missing,
        "forbidden_pattern_results": forbidden_hits,
        "functional_checks": checks,
        "smoke_result": {**smoke_result, "uses_real_dino": (args.device == "cuda" and torch.cuda.is_available())},
        "warnings": [],
        "review_pass_path": str(out / "REVIEW_PASS_ACPR_PACE_V1.txt"),
    }
    (out / "implementation_audit_ACPR_PACE_V1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if pass_flag and args.write_review_pass:
        (out / "REVIEW_PASS_ACPR_PACE_V1.txt").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if pass_flag else 1)


if __name__ == "__main__":
    main()
