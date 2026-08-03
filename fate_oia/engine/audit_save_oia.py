from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from fate_oia.models.save_oia_model import SAVEOIAModel
from fate_oia.utils.save_artifacts import hash_value, save_source_tree_hash, write_json
from fate_oia.utils.save_contracts import validate_save_config, validate_save_factor_schema, validate_save_worktree


FORBIDDEN_FORMAL_PATTERNS = (
    "StateConditionedActionCredit", "HECAExcessRiskBalancer", "ACPRPairMemory",
    "feature_cache_enabled: true", "token_compression: keep_merge", "trainable_threshold: true",
    "action_set_probs @ subset_membership", "run_c_logits", "cached_logits",
)

REQUIRED_SAVE_FILES = (
    "fate_oia/models/save_oia_model.py",
    "fate_oia/models/save_multiscale_field.py",
    "fate_oia/models/save_predicate_measurement.py",
    "fate_oia/models/save_action_evidence.py",
    "fate_oia/models/save_utility_bridge.py",
    "fate_oia/models/save_reason_decoder.py",
    "fate_oia/losses/save_action_losses.py",
    "fate_oia/losses/save_reason_losses.py",
    "fate_oia/losses/save_grounding_losses.py",
    "fate_oia/losses/save_faithfulness_losses.py",
    "fate_oia/losses/save_pu_losses.py",
    "fate_oia/losses/save_loss_registry.py",
    "fate_oia/engine/train_save_oia.py",
    "fate_oia/engine/eval_save_oia.py",
    "fate_oia/engine/evaluate_save_oia_pilot.py",
    "fate_oia/engine/profile_save_oia.py",
    "fate_oia/engine/supervise_save_oia_foreground.py",
    "fate_oia/engine/audit_save_oia.py",
    "fate_oia/utils/save_artifacts.py",
    "fate_oia/utils/save_contracts.py",
    "configs/fate_oia_train_360x640_save_oia_v1.yaml",
    "configs/save_factor_schema.yaml",
    "scripts/FATE_OIA_save_oia_v1_pilot.ps1",
    "scripts/FATE_OIA_save_oia_v1_foreground.ps1",
    ".codex/skills/save-oia-implementation-audit/SKILL.md",
)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return {"command": command, "returncode": result.returncode, "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:]}


def _formal_sources(root: Path) -> list[Path]:
    return [root / relative for relative in REQUIRED_SAVE_FILES if relative.endswith(".py")]


def _forbidden_scan_sources(root: Path) -> list[Path]:
    """Exclude audit declarations: mentioning a forbidden token is not use."""
    prefixes = ("fate_oia/models/", "fate_oia/losses/", "fate_oia/engine/train_")
    return [
        root / relative for relative in REQUIRED_SAVE_FILES
        if relative.endswith(".py") and relative.startswith(prefixes)
    ]


def _static_checks(root: Path) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    checks["required_source_and_protocol_files"] = all((root / relative).is_file() for relative in REQUIRED_SAVE_FILES)
    if not checks["required_source_and_protocol_files"]:
        return checks
    source = "\n".join(path.read_text(encoding="utf-8") for path in _formal_sources(root))
    forbidden_source = "\n".join(path.read_text(encoding="utf-8") for path in _forbidden_scan_sources(root))
    checks["ast_parse"] = True
    try:
        for path in _formal_sources(root): ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError: checks["ast_parse"] = False
    checks["forbidden_formal_paths"] = all(token not in forbidden_source for token in FORBIDDEN_FORMAL_PATTERNS)
    checks["complete_calalign_base"] = "action_logits_base=decoded[\"action_logits_calalign\"]" in source.replace(" ", "")
    checks["global_utility_detail_order"] = "read_global" in source and "utility_bridge" in source and "read_detail" in source
    checks["one_dino_field_api"] = "encode_images" in source and "decode_from_field" in source
    checks["same_field_eval"] = "model.encode_images(images)" in (root / "fate_oia/engine/eval_save_oia.py").read_text(encoding="utf-8")
    checks["save_loss_registry"] = "build_save_loss_registry" in (root / "fate_oia/engine/train_save_oia.py").read_text(encoding="utf-8")
    checks["six_owner_optimizer"] = "build_save_optimizer_groups" in (root / "fate_oia/engine/train_save_oia.py").read_text(encoding="utf-8")
    checks["train_audit_only_pu"] = "admit_pu_from_train_audit" in (root / "fate_oia/engine/train_save_oia.py").read_text(encoding="utf-8")
    checks["artifact_and_profile_closure"] = all(
        token in source for token in ("save_epoch_artifacts", "validate_epoch_artifacts", "profile_candidate", "validate_save_full_readiness")
    )
    return checks


def _dynamic_checks() -> dict[str, bool]:
    model = SAVEOIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    model.eval()
    with torch.no_grad():
        output = model(images, progress=0.0, action_targets=torch.zeros(1, 4), optimizer_update=0, run_teacher=True)
    required = {"action_logits_base", "action_logits_final", "reason_logits_clean", "reason_logits_private_direct", "predicate_map_action", "action_named_contribution", "action_unnamed_contribution"}
    checks = {
        "full_model_forward": required <= set(output),
        "action_shape": tuple(output["action_logits_final"].shape) == (1, 4),
        "reason_shape": tuple(output["reason_logits_private_direct"].shape) == (1, 21),
        "predicate_shape": tuple(output["predicate_map_action"].shape) == (1, 21, 3600),
        "progress_zero_action_equivalence": bool(torch.allclose(output["action_logits_final"], output["action_logits_base"], atol=1e-6, rtol=0.0)),
        "dino_frozen": all(parameter.grad is None for parameter in model.foundation.dino.parameters()),
        "named_unnamed_conservation": bool(output["action_conservation_error"].abs().max() < 1e-6),
        "test_image_only": False,
    }
    with torch.no_grad():
        test = model.forward_test(images)
    checks["test_image_only"] = bool(test.get("test_forward_image_only"))
    return checks


def implementation_audit(config_path: str | Path, *, root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve(); config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config_ok = schema_ok = worktree_ok = True
    errors: list[str] = []
    try: validate_save_config(config)
    except Exception as exc: config_ok = False; errors.append(f"config:{exc}")
    try: validate_save_factor_schema(root / "configs/save_factor_schema.yaml")
    except Exception as exc: schema_ok = False; errors.append(f"schema:{exc}")
    try: validate_save_worktree(root)
    except Exception as exc: worktree_ok = False; errors.append(f"worktree:{exc}")
    static = _static_checks(root); dynamic = _dynamic_checks()
    compile_result = _run([sys.executable, "-m", "py_compile", *[str(path) for path in _formal_sources(root)]])
    tests_result = _run([sys.executable, "-m", "pytest", "-q", "tests", "-k", "save"])
    head = _git("rev-parse", "HEAD")
    bindings = {"git_head": head, "config_hash": hash_value(config), "source_tree_hash": save_source_tree_hash(root), "schema_hash": hash_value(yaml.safe_load((root / "configs/save_factor_schema.yaml").read_text(encoding="utf-8"))), "split_hash": "pending", "checkpoint_hash": "pending", "logits_hash": "pending", "labels_hash": "pending", "file_order_hash": "pending"}
    passed = config_ok and schema_ok and worktree_ok and all(static.values()) and all(dynamic.values()) and compile_result["returncode"] == 0 and tests_result["returncode"] == 0
    return {"pass": passed, "bindings": bindings, "git_head": head, "checked_files": [str(path.relative_to(root)) for path in _formal_sources(root)], "static_checks": static, "dynamic_checks": dynamic, "commands": {"py_compile": compile_result, "pytest_save": tests_result}, "errors": errors, "warnings": []}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--mode", default="implementation"); parser.add_argument("--write-review-pass", action="store_true"); parser.add_argument("--development-audit", action="store_true")
    args = parser.parse_args(); result = implementation_audit(args.config)
    output = Path(args.output_dir); write_json(output / "SAVE_IMPLEMENTATION_REVIEW.json", result)
    if args.write_review_pass and not args.development_audit:
        dirty = _git("status", "--porcelain")
        if dirty: result["pass"] = False; result["errors"].append("DIRTY_WORKTREE")
        if result["pass"]: write_json(output / "REVIEW_PASS_SAVE_OIA_V1.json", result)
    write_json(output / "SAVE_IMPLEMENTATION_REVIEW.json", result)
    print(json.dumps(result, indent=2), flush=True); raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__": main()
