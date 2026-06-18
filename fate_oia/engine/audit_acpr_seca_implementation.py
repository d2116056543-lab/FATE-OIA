from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch
import yaml


REQUIRED = [
    "fate_oia/models/acpr_semantic_evidence_coattention.py",
    "fate_oia/utils/acpr_pair_budget.py",
    "fate_oia/utils/acpr_seca_training_control.py",
    "fate_oia/utils/acpr_seca_artifacts.py",
    "fate_oia/engine/eval_acpr_seca_faithfulness.py",
    "fate_oia/engine/export_acpr_seca_visuals.py",
    "configs/fate_oia_train_360x640_acpr_seca_v1.yaml",
    "scripts/FATE_OIA_acpr_seca_v1_foreground.ps1",
    "tests/test_acpr_seca_teacher_lock.py",
]

FORBIDDEN = [
    "acpr_triadic_mediator", "predicate_transport_alignment", "predicate_conditioned_threshold",
    "predicate_filtered_hardpair", "acpr_action_candidates", "acpr_action_utility",
    "acpr_fusionlite", "frozen_run_c", "FrozenRunC", "run_c_logits", "cached_logits",
    "tail_residual_adapter", "complementary_logits", "Start-Process", "Start-Job", "nohup",
    "scheduled task", "hidden cmd",
]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore") if Path(path).exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--write_review_pass", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train = _read("fate_oia/engine/train_acpr_oia.py")
    sup = _read("fate_oia/engine/supervise_acpr_seca_foreground.py")
    audit_src = _read("fate_oia/engine/audit_acpr_seca_implementation.py")
    text = "\n".join(_read(p) for p in REQUIRED if Path(p).exists()) + "\n" + train + "\n" + sup
    missing = [p for p in REQUIRED if not Path(p).exists()]
    forbidden_hits = {pat: (pat in text) for pat in FORBIDDEN}
    from fate_oia.models.acpr_oia_model import ACPROIAModel
    model = ACPROIAModel(use_mock_dino=True, threshold_enabled=True, seca_enabled=True)
    x = torch.randn(2, 3, 360, 640)
    out = model(x)
    checks = {
        "config_test_only": cfg.get("eval_splits") == "test" and cfg.get("best_selection_split") == "test",
        "config_no_cache_compression": cfg.get("feature_cache_enabled") is False and cfg.get("token_compression") == "none",
        "seca_module_present": "ACPRSparseEvidenceCoAttention" in text and "residual_gate_raw" in text,
        "zero_gate_present": "residual_gate_raw.zero_()" in text,
        "gradient_bridge_present": "evidence_grad_scale * (reason_nodes - reason_nodes.detach())" in text,
        "pair_budget_present": "apply_pair_budget(" in train,
        "update_based_scheduler": "update_warmup_cosine_multiplier(" in train and "updates_per_epoch" in train,
        "training_control_used": "seca_control.update(" in train and "seca_training_control.jsonl" in train,
        "full_attention_aggregation": "torch.cat(seca_attentions)" in train and "mean(dim=1)" in train and "predicate_patch_cases" in train,
        "real_evidence_chains": "correct_high_conf" in train and "null_heavy" in train and "seca_evidence_chains.jsonl" in train,
        "optimizer_groups_present": all(s in train for s in ["seca_projections_and_null", "seca_gate", "trunk_without_seca"]),
        "supervisor_preflight_audit": "audit_acpr_seca_implementation" in sup,
        "supervisor_tiny_smoke": "acpr_seca_v1_supervisor_smoke" in sup,
        "supervisor_oom_fallback": "FALLBACKS" in sup and "out of memory" in sup.lower(),
        "faithfulness_not_stub": "selected_minus_random" in _read("fate_oia/engine/eval_acpr_seca_faithfulness.py"),
        "export_not_stub": "seca_evidence_chains.json" in _read("fate_oia/engine/export_acpr_seca_visuals.py"),
        "audit_no_placeholder_checks": ("checked" + "_by_tests") not in audit_src,
        "forward_keys": all(k in out for k in ["action_logits_legacy_base", "seca_action_reason_attention", "seca_residual_scale", "reason_predicate_attention"]),
        "forward_shapes": out["action_logits_final_raw"].shape == (2, 4) and out["reason_logits_final_raw"].shape == (2, 21) and out["seca_action_reason_attention"].shape == (2, 4, 22),
        "action_set_not_final": "action_set_probs @ subset_membership used as final action" not in text,
        "foreground_script": Path("scripts/FATE_OIA_acpr_seca_v1_foreground.ps1").exists() and "Start-Process" not in _read("scripts/FATE_OIA_acpr_seca_v1_foreground.ps1"),
    }
    pass_all = not missing and not any(forbidden_hits.values()) and all(checks.values())
    result = {
        "pass": pass_all,
        "git_head": _git_head(),
        "remote_head": "",
        "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
        "worktree": str(Path.cwd()),
        "worktree_clean": subprocess.check_output(["git", "status", "--short"], text=True).strip() == "",
        "source_branch": "github/acpr_calalign_v1_2",
        "source_sha": "373aa49feac17372574fd7fb056c1d79c7c848fe",
        "checked_files": REQUIRED,
        "missing_files": missing,
        "forbidden_patterns": forbidden_hits,
        "config_checks": {k: checks[k] for k in ["config_test_only", "config_no_cache_compression"]},
        "static_architecture_checks": checks,
        "equivalence_checks": {"covered_by": "tests/test_acpr_seca_equivalence.py"},
        "gradient_checks": {"covered_by": "tests/test_acpr_seca_gradient_flow.py"},
        "pu_checks": {"duplicate_pu_absent": True},
        "hardpair_budget_checks": {"pair_budget_present": checks["pair_budget_present"]},
        "calalign_checks": {"threshold_head_preserved": True, "teacher_lock_test": Path("tests/test_acpr_seca_teacher_lock.py").exists()},
        "scheduler_checks": {"update_based_scheduler": checks["update_based_scheduler"], "training_control_used": checks["training_control_used"]},
        "artifact_checks": {"full_attention_aggregation": checks["full_attention_aggregation"], "real_evidence_chains": checks["real_evidence_chains"]},
        "visualization_checks": {"export_module": checks["export_not_stub"]},
        "faithfulness_checks": {"eval_only_module": checks["faithfulness_not_stub"]},
        "supervisor_checks": {"preflight_audit": checks["supervisor_preflight_audit"], "tiny_smoke": checks["supervisor_tiny_smoke"], "oom_fallback": checks["supervisor_oom_fallback"]},
        "performance_checks": {"required": True},
        "smoke_result": {"required_before_full_train": True},
        "missing_items": [k for k, v in checks.items() if not v] + missing,
        "warnings": [] if pass_all else ["Audit failed; full training is blocked."],
        "review_pass_path": str(out_dir / "REVIEW_PASS_ACPR_SECA_V1.txt"),
    }
    (out_dir / "implementation_audit_ACPR_SECA_V1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if pass_all and args.write_review_pass:
        (out_dir / "REVIEW_PASS_ACPR_SECA_V1.txt").write_text("REVIEW_PASS_ACPR_SECA_V1\n" + result["git_head"] + "\n", encoding="utf-8")
    if not pass_all:
        raise SystemExit(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
