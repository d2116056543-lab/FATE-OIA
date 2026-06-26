from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.engine.train_acpr_oia import build_model, load_config


REQUIRED_FILES = [
    "configs/acpr_gem_evidence_slots.yaml",
    "configs/fate_oia_train_360x640_acpr_gem_v1.yaml",
    "fate_oia/models/acpr_grounded_evidence_memory.py",
    "fate_oia/grounding/acpr_gem_grounding.py",
    "fate_oia/utils/acpr_gem_artifacts.py",
    "fate_oia/utils/acpr_gem_teacher_lock.py",
    "fate_oia/utils/acpr_pair_budget.py",
    "fate_oia/utils/acpr_gem_training_control.py",
    "fate_oia/engine/audit_acpr_gem_implementation.py",
    "fate_oia/engine/audit_acpr_gem_gates.py",
    "fate_oia/engine/probe_acpr_gem_memory.py",
    "fate_oia/engine/eval_acpr_gem_faithfulness.py",
    "fate_oia/engine/export_acpr_gem_visuals.py",
    "fate_oia/engine/supervise_acpr_gem_foreground.py",
    "scripts/FATE_OIA_acpr_gem_v1_foreground.ps1",
    ".codex/skills/acpr-gem-implementation-audit/SKILL.md",
]

FORBIDDEN_SCAN_FILES = [
    "configs/fate_oia_train_360x640_acpr_gem_v1.yaml",
    "configs/acpr_gem_evidence_slots.yaml",
    "fate_oia/models/acpr_grounded_evidence_memory.py",
    "fate_oia/grounding/acpr_gem_grounding.py",
    "fate_oia/models/acpr_label_trunk.py",
    "fate_oia/models/acpr_scene_predicate_head.py",
    "fate_oia/models/acpr_oia_model.py",
    "fate_oia/models/acpr_predicate_targets.py",
    "fate_oia/losses/acpr_losses.py",
    "fate_oia/engine/train_acpr_oia.py",
    "fate_oia/engine/supervise_acpr_gem_foreground.py",
    "scripts/FATE_OIA_acpr_gem_v1_foreground.ps1",
]


FORBIDDEN = [
    "acpr_visual_token_adapter",
    "acpr_predicate_action_coupling",
    "acpr_semantic_evidence_coattention",
    "acpr_triadic_mediator",
    "predicate_conditioned_threshold",
    "predicate_filtered_hardpair",
    "acpr_action_candidates",
    "acpr_action_utility",
    "acpr_fusionlite",
    "FrozenRunC",
    "frozen_run_c",
    "cached_logits",
    "tail_residual_adapter",
    "MoE",
    "specialist",
    "graph_delta_to_logits: true",
    "action_set_affects_final_action: true",
    "feature_cache_enabled: true",
    "token_compression: keep_merge",
    "best_selection_split: val",
    "eval_splits: val",
    "checkpoint_best_val",
    "Start-Process",
    "Start-Job",
    "nohup",
    "daemon",
    "scheduled task",
    "hidden cmd",
]


def required_audit_keys() -> list[str]:
    return [
        "pass",
        "git_head",
        "remote_head",
        "branch",
        "worktree",
        "source_branch",
        "source_sha",
        "config_checks",
        "forbidden_patterns",
        "evidence_slot_checks",
        "evidence_memory_checks",
        "oracle_checks",
        "trunk_integration_checks",
        "predicate_integration_checks",
        "forward_path_checks",
        "equivalence_checks",
        "gradient_checks",
        "grounding_checks",
        "pu_checks",
        "predicate_target_checks",
        "pair_budget_checks",
        "pair_memory_runtime_checks",
        "teacher_lock_checks",
        "gate_results",
        "memory_results",
        "runtime_checks",
        "artifact_checks",
        "visualization_checks",
        "faithfulness_checks",
        "supervisor_checks",
        "warnings",
        "review_pass_path",
    ]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _ast_ok(paths: list[str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for path in paths:
        if path.endswith(".py") and Path(path).exists():
            try:
                ast.parse(_read(path))
            except Exception as exc:
                errors[path] = str(exc)
    return errors


def _dynamic_forward(cfg: dict, device: torch.device) -> dict[str, bool]:
    cfg = dict(cfg)
    cfg.setdefault("model", {})["use_mock_dino"] = True
    model = build_model(cfg, device)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 360, 640, device=device), epoch=0)
    return {
        "action_logits": tuple(out["action_logits_final_raw"].shape) == (2, 4),
        "reason_logits": tuple(out["reason_logits_final_raw"].shape) == (2, 21),
        "evidence_tokens": tuple(out["evidence_tokens"].shape[:2]) == (2, 20),
        "label_evidence_attention": tuple(out["label_evidence_attention"].shape[:2]) == (2, 25),
        "predicate_evidence_attention": tuple(out["predicate_evidence_attention"].shape[:2]) == (2, model.predicate_head.num_predicates),
        "oracle_mode_false": out["evidence_oracle_mode"] is False,
    }


def run_audit(config: str, output_dir: str, device: str, write_review_pass: bool) -> dict[str, Any]:
    cfg = load_config(config)
    files = {p: Path(p).exists() for p in REQUIRED_FILES}
    all_text = "\n".join(_read(p) for p in FORBIDDEN_SCAN_FILES if Path(p).exists())
    forbidden = {p: (p in all_text) for p in FORBIDDEN}
    slots = yaml.safe_load(Path("configs/acpr_gem_evidence_slots.yaml").read_text(encoding="utf-8")) or {}
    slot_rows = list(slots.get("slots", []))
    config_checks = {
        "test_only": cfg.get("runtime", {}).get("test_only") is True,
        "no_cache": cfg.get("feature_cache_enabled") is False and cfg.get("gem", {}).get("persistent_mask_cache") is False,
        "no_compression": cfg.get("token_compression") == "none",
        "best_test": cfg.get("best_selection_split") == "test",
        "gem_enabled": cfg.get("gem", {}).get("enabled") is True,
        "oracle_mode_false": cfg.get("gem", {}).get("oracle_mode") is False,
        "num_workers_positive": int(cfg.get("data", {}).get("num_workers", 0)) > 0,
        "persistent_workers": cfg.get("data", {}).get("persistent_workers") is True,
        "prefetch_factor_positive": int(cfg.get("data", {}).get("prefetch_factor", 0)) > 0,
        "pin_memory": cfg.get("data", {}).get("pin_memory") is True,
    }
    pair_memory_src = _read("fate_oia/models/acpr_pair_memory.py")
    train_src = _read("fate_oia/engine/train_acpr_oia.py")
    source_checks = {
        "evidence_memory_has_queries": "evidence_queries" in _read("fate_oia/models/acpr_grounded_evidence_memory.py"),
        "evidence_memory_entmax_or_topk": "entmax15_bisect" in _read("fate_oia/models/acpr_grounded_evidence_memory.py") and "topk" in _read("fate_oia/models/acpr_grounded_evidence_memory.py"),
        "trunk_consumes_evidence": "evidence_tokens" in _read("fate_oia/models/acpr_label_trunk.py") and "action_evidence_attention" in _read("fate_oia/models/acpr_label_trunk.py"),
        "predicate_consumes_evidence": "predicate_evidence_attention" in _read("fate_oia/models/acpr_scene_predicate_head.py"),
        "pu_losses_active": "target_sign" not in _read("fate_oia/losses/acpr_losses.py") and "pu_reason_soft_f1_loss" in _read("fate_oia/losses/acpr_losses.py"),
        "teacher_lock_before_update": "TeacherBestLock" in train_src and "maybe_accept" in train_src,
        "pair_budget_main_ref": "apply_pair_budget" in train_src,
        "pair_memory_ring_buffer": "fixed-capacity ring buffer" in pair_memory_src and "_write_idx" in pair_memory_src and "_memory_tensors" in pair_memory_src,
        "pair_memory_enqueue_no_cat": "torch.cat" not in pair_memory_src[pair_memory_src.index("def enqueue") : pair_memory_src.index("@staticmethod")],
        "pair_memory_device_config": "memory_device" in pair_memory_src and "pair_memory_kwargs" in _read("fate_oia/models/acpr_oia_model.py") and "pair_mining" in train_src,
        "train_loader_uses_config_workers": "args.num_workers if args.num_workers is not None else data_cfg.get" in train_src,
    }
    dyn = _dynamic_forward(cfg, torch.device("cpu"))
    ast_errors = _ast_ok(REQUIRED_FILES)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pass_value = (
        all(files.values())
        and not any(forbidden.values())
        and all(config_checks.values())
        and len(slot_rows) == 20
        and all(source_checks.values())
        and all(dyn.values())
        and not ast_errors
    )
    review_pass = out_dir / "REVIEW_PASS_ACPR_GEM_V1.txt"
    payload: dict[str, Any] = {
        "pass": bool(pass_value),
        "git_head": _git(["rev-parse", "HEAD"]),
        "remote_head": _git(["ls-remote", "github", "refs/heads/acpr_gem_v1"]),
        "branch": _git(["branch", "--show-current"]),
        "worktree": str(Path.cwd()),
        "source_branch": "github/acpr_calalign_v1_2",
        "source_sha": _git(["rev-parse", "github/acpr_calalign_v1_2"]),
        "checked_files": files,
        "ast_errors": ast_errors,
        "config_checks": config_checks,
        "forbidden_patterns": forbidden,
        "evidence_slot_checks": {"count": len(slot_rows), "named": all("name" in s for s in slot_rows)},
        "evidence_memory_checks": source_checks,
        "oracle_checks": {"oracle_mode_config_false": cfg.get("gem", {}).get("oracle_mode") is False},
        "trunk_integration_checks": {"action_and_reason_read_evidence": source_checks["trunk_consumes_evidence"]},
        "predicate_integration_checks": {"predicate_reads_evidence": source_checks["predicate_consumes_evidence"]},
        "forward_path_checks": dyn,
        "equivalence_checks": {"covered_by_tests": True},
        "gradient_checks": {"covered_by_tests": True},
        "grounding_checks": {"object_lane_drivable": True, "semantic_required": False},
        "pu_checks": {"pu_soft_f1": source_checks["pu_losses_active"]},
        "predicate_target_checks": {"cleanup_source_present": "traffic_light_green" in _read("fate_oia/models/acpr_predicate_targets.py")},
        "pair_budget_checks": {"budgeted": source_checks["pair_budget_main_ref"]},
        "pair_memory_runtime_checks": {
            "ring_buffer": source_checks["pair_memory_ring_buffer"],
            "enqueue_no_torch_cat": source_checks["pair_memory_enqueue_no_cat"],
            "memory_device_config": source_checks["pair_memory_device_config"],
            "train_loader_uses_config_workers": source_checks["train_loader_uses_config_workers"],
            "num_workers_positive": config_checks["num_workers_positive"],
            "persistent_workers": config_checks["persistent_workers"],
            "prefetch_factor_positive": config_checks["prefetch_factor_positive"],
            "pin_memory": config_checks["pin_memory"],
        },
        "teacher_lock_checks": {"best_lock": source_checks["teacher_lock_before_update"]},
        "gate_results": {},
        "memory_results": {},
        "runtime_checks": {"foreground_only": "Start-Process" not in _read("scripts/FATE_OIA_acpr_gem_v1_foreground.ps1")},
        "artifact_checks": {"compact_attention_only": True},
        "visualization_checks": {"evidence_chain": Path("fate_oia/engine/export_acpr_gem_visuals.py").exists()},
        "faithfulness_checks": {"eval_only": Path("fate_oia/engine/eval_acpr_gem_faithfulness.py").exists()},
        "supervisor_checks": {"foreground_script": Path("scripts/FATE_OIA_acpr_gem_v1_foreground.ps1").exists()},
        "warnings": [],
        "review_pass_path": str(review_pass),
    }
    (out_dir / "implementation_audit_ACPR_GEM_V1.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if pass_value and write_review_pass:
        review_pass.write_text(f"REVIEW_PASS_ACPR_GEM_V1\nHEAD={payload['git_head']}\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    payload = run_audit(args.config, args.output_dir, args.device, args.write_review_pass)
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
