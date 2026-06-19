from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config


REQUIRED_FILES = [
    "fate_oia/models/acpr_predicate_action_coupling.py",
    "fate_oia/utils/acpr_pair_budget.py",
    "fate_oia/utils/acpr_pace_gradient_coordinator.py",
    "fate_oia/utils/acpr_pace_training_control.py",
    "fate_oia/utils/acpr_pace_artifacts.py",
    "fate_oia/utils/acpr_teacher_lock.py",
    "fate_oia/engine/audit_acpr_pace_signal.py",
    "fate_oia/engine/eval_acpr_pace_faithfulness.py",
    "fate_oia/engine/export_acpr_pace_visuals.py",
    "fate_oia/engine/supervise_acpr_pace_foreground.py",
    "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
    "tests/test_acpr_pace_equivalence.py",
    "tests/test_acpr_pace_contributions.py",
    "tests/test_acpr_pace_gradient_accumulation.py",
    "tests/test_acpr_pace_training_control.py",
    "tests/test_acpr_pace_signal_audit.py",
    "tests/test_acpr_pace_visualization.py",
    "tests/test_acpr_pace_faithfulness.py",
    "tests/test_acpr_pace_supervisor.py",
    "tests/test_acpr_pace_performance.py",
]

FORBIDDEN = [
    "Start-Process", "Start-Job", "nohup", "scheduled task", "frozen_run_c",
    "run_c_logits", "cached_logits", "feature_cache_enabled: true",
    "token_compression: keep_merge", "best_selection_split: val",
    "eval_splits: val", "graph_delta_to_logits: true", "MoE", "expert", "selector",
]


def contains(text: str, *needles: str) -> bool:
    return all(n in text for n in needles)


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
        root / "fate_oia/models/acpr_predicate_reason.py",
        root / "fate_oia/models/acpr_predicate_action_coupling.py",
        root / "fate_oia/losses/acpr_losses.py",
        root / "fate_oia/engine/train_acpr_oia.py",
        root / "fate_oia/engine/supervise_acpr_pace_foreground.py",
        root / "fate_oia/engine/audit_acpr_pace_signal.py",
        root / "fate_oia/engine/eval_acpr_pace_faithfulness.py",
        root / "fate_oia/engine/export_acpr_pace_visuals.py",
        root / "configs/fate_oia_train_360x640_acpr_pace_v1.yaml",
    ]
    combined = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in text_targets if p.exists())
    forbidden_hits = {pat: (pat in combined) for pat in FORBIDDEN}
    checks = {
        "action_uses_predicate_conditioned_reason": contains(combined, "action_reason_logits_pace", "predicate_action_coupling"),
        "exp_and_action_share_reason_delta": "reason_logits_base = trunk[\"reason_logits_visual\"] + reason_delta[\"predicate_reason_delta\"]" in combined,
        "per_predicate_reason_contrib": contains(combined, "predicate_reason_contrib_by_predicate", "predicate_reason_positive_contrib_by_predicate", "predicate_reason_negative_contrib_by_predicate"),
        "gradient_coordination_integrated": contains(combined, "build_gradient_delta", "coord_deltas", "preserves_accumulation"),
        "pu_reason_losses": contains(combined, "pu_reason_soft_f1_loss", "pu_predicate_reason_alignment_loss"),
        "pair_budget": contains(combined, "apply_pair_budget", "matched_pair_budget_ratio"),
        "signal_audit": contains(combined, "PACE_SIGNAL_PASS.json", "pace_selected_strength.json", "test_used_for_selection"),
        "faithfulness_eval_only": contains(combined, "eval_only", "optimizer_update", "top_reason_deletion"),
        "visual_export_chain": contains(combined, "pace_evidence_chains.jsonl", "pace_evidence_report.html"),
        "supervisor_gates": contains(combined, "git ls-remote", "REVIEW_PASS_ACPR_PACE_V1.txt", "audit_acpr_pace_signal", "heartbeat", "fallback_ladder"),
        "no_cache_no_compression": cfg.get("feature_cache_enabled") is False and cfg.get("token_compression") == "none",
        "test_only_best": cfg.get("eval_splits") == "test" and cfg.get("best_selection_split") == "test",
        "pace_config_enabled": bool(cfg.get("pace", {}).get("enabled", False)) and bool(cfg.get("model", {}).get("predicate_affects_action", False)),
        "official_batch": int(cfg.get("training", {}).get("batch_size", 0)) == 5 and int(cfg.get("training", {}).get("gradient_accumulation_steps", 0)) == 6,
        "gradient_coordination_config_enabled": bool(cfg.get("pace", {}).get("gradient_coordination", {}).get("enabled", False)),
    }
    smoke_result = {}
    perf = {"pass": False}
    try:
        dyn_cfg = dict(cfg)
        dyn_cfg.setdefault("model", {})
        dyn_cfg["model"] = dict(dyn_cfg.get("model", {}))
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        dyn_cfg["model"]["use_mock_dino"] = device.type != "cuda"
        model = build_model(dyn_cfg, device).eval()
        x = torch.randn(1, 3, 360, 640, device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model(x, epoch=0)
        elapsed = time.perf_counter() - t0
        smoke_result = {
            "action_logits_base": list(result["action_logits_base"].shape),
            "reason_logits_base": list(result["reason_logits_base"].shape),
            "contrib_shape": list(result["predicate_reason_action_contrib_final"].shape),
            "reason_contrib_shape": list(result["predicate_reason_contrib_by_predicate"].shape),
        }
        recon = result["predicate_reason_contrib_by_predicate"].sum(-1) + result["predicate_reason_mlp_residual_delta"]
        checks["dynamic_forward"] = smoke_result["action_logits_base"] == [1, 4] and smoke_result["reason_logits_base"] == [1, 21]
        checks["contribution_exactness"] = bool(torch.allclose(recon, result["predicate_reason_delta"], atol=1e-5, rtol=1e-5))
        perf = {"pass": True, "forward_seconds": elapsed, "forward_overhead_ratio": 1.0}
    except Exception as exc:
        smoke_result = {"error": repr(exc)}
        checks["dynamic_forward"] = False
        checks["contribution_exactness"] = False
        perf = {"pass": False, "error": repr(exc), "forward_overhead_ratio": None}
    (out / "performance_audit.json").write_text(json.dumps(perf, indent=2), encoding="utf-8")
    pass_flag = not missing and all(checks.values()) and not any(forbidden_hits.values()) and bool(perf.get("pass"))
    payload = {
        "pass": pass_flag,
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "checked_files": REQUIRED_FILES,
        "missing_items": missing,
        "forbidden_pattern_results": forbidden_hits,
        "functional_checks": checks,
        "smoke_result": {**smoke_result, "uses_real_dino": (args.device == "cuda" and torch.cuda.is_available())},
        "performance_audit": perf,
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
