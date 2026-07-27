"""Fail-closed P21 audit for the RAEL-OIA implementation and launch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REQUIRED_SOURCE_FILES = (
    "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml",
    "configs/rael_reason_semantics.yaml",
    "configs/rael_slot_schema.yaml",
    "configs/rael_action_semantics.yaml",
    "fate_oia/engine/train_acpr_rael_oia.py",
    "fate_oia/engine/eval_acpr_rael_oia.py",
    "fate_oia/datasets/bdd100k_task_aware_index.py",
    "fate_oia/datasets/rael_grounding_targets.py",
    "fate_oia/transforms_rael.py",
    "fate_oia/models/rael_dino_field.py",
    "fate_oia/models/rael_multilayer_field.py",
    "fate_oia/models/rael_slot_ledger.py",
    "fate_oia/models/rael_semantic_reason.py",
    "fate_oia/models/rael_category_foundation.py",
    "fate_oia/models/rael_action_reason_bridge.py",
    "fate_oia/models/rael_relation_contributions.py",
    "fate_oia/models/rael_reason_private.py",
    "fate_oia/models/rael_oia_model.py",
    "fate_oia/losses/rael_task_losses.py",
    "fate_oia/losses/rael_grounding_losses.py",
    "fate_oia/losses/rael_counterfactual_losses.py",
    "fate_oia/losses/rael_pu_losses.py",
    "fate_oia/optim/rael_gradient_admission.py",
    "fate_oia/engine/export_rael_cases.py",
    "fate_oia/engine/profile_acpr_rael_oia.py",
    "fate_oia/engine/audit_acpr_rael_oia.py",
    "fate_oia/engine/supervise_acpr_rael_oia_foreground.py",
    "fate_oia/utils/rael_artifacts.py",
    "fate_oia/utils/rael_runtime.py",
    "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1",
    ".codex/skills/rael-oia-v1-implementation-audit/SKILL.md",
)
REQUIRED_TEST_FILES = (
    "tests/test_rael_worktree_contract.py",
    "tests/test_rael_reason_schema.py",
    "tests/test_rael_grounding_index.py",
    "tests/test_rael_dino_contract.py",
    "tests/test_rael_multilayer_reading.py",
    "tests/test_rael_slot_competition.py",
    "tests/test_rael_slot_attributes.py",
    "tests/test_rael_absence_evidence.py",
    "tests/test_rael_semantic_reason.py",
    "tests/test_rael_action_reason_firewall.py",
    "tests/test_rael_adaptive_entmax.py",
    "tests/test_rael_unary_contribution.py",
    "tests/test_rael_pairwise_relation.py",
    "tests/test_rael_reason_private.py",
    "tests/test_rael_pu.py",
    "tests/test_rael_gradient_admission.py",
    "tests/test_rael_counterfactual.py",
    "tests/test_rael_posthoc_calibration.py",
    "tests/test_rael_model_forward.py",
    "tests/test_rael_train_protocol.py",
    "tests/test_rael_trainer_handoff.py",
    "tests/test_rael_eval_contract.py",
    "tests/test_rael_runtime.py",
    "tests/test_rael_artifacts.py",
    "tests/test_rael_supervisor.py",
    "tests/test_rael_audit.py",
)
FORBIDDEN_PATTERNS = (
    "Start-Process",
    "Start-Job",
    "Register-ScheduledTask",
    "-WindowStyle Hidden",
    "nohup",
    "scheduled task",
    "daemon",
    "cached_logits",
    "run_c_checkpoint",
    "token_compression: keep_merge",
    "feature_cache_enabled: true",
    "best_selection_split: val",
    "ACPROIAModel",
    "ACPRLabelTrunk",
    "ACPRScenePredicateHead",
    "ACPRPredicateReasoner",
    "ACPRPairMemory",
    "ACPRActionComboAux",
    "ACPRCalibrationHead",
    "ACPRThresholdHead",
)
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    )
    head = result.stdout.strip()
    if not _SHA.fullmatch(head):
        raise ValueError("git HEAD is not a full lowercase SHA")
    return head


def pilot_override_payload(*, enabled: bool, reason: str) -> dict[str, Any]:
    if not enabled or not isinstance(reason, str) or not reason.strip():
        raise ValueError("a pilot override must be explicit and include a nonempty user reason")
    return {
        "pilot_protocol_override": True,
        "pilot_completed": False,
        "reason": reason.strip(),
        "replacement": "minimal_real_smoke_only",
    }


def _required_files(root: Path, names: Sequence[str]) -> dict[str, bool]:
    return {name: (root / name).is_file() for name in names}


def _forbidden_results(root: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {pattern: [] for pattern in FORBIDDEN_PATTERNS}
    # Audit code and tests intentionally spell forbidden strings, so scanning
    # them would make every audit self-fail. Scan only executable formal paths.
    candidates = [
        root / "fate_oia/models/rael_oia_model.py",
        root / "fate_oia/models/rael_dino_field.py",
        root / "fate_oia/engine/supervise_acpr_rael_oia_foreground.py",
        root / "fate_oia/engine/train_acpr_rael_oia.py",
        root / "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        relative = path.relative_to(root).as_posix()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                results[pattern].append(relative)
    return results


def validate_explicit_grounding_layout(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the host-declared per-frame train/val layout exists."""

    sources = config.get("grounding_sources")
    if not isinstance(sources, Mapping):
        raise ValueError("grounding_sources is required; aggregate-path guessing is forbidden")
    directories = sources.get("label_directories")
    if not isinstance(directories, Mapping) or set(directories) != {"train", "val"}:
        raise ValueError("grounding_sources.label_directories must contain exactly train and val")
    resolved: dict[str, str] = {}
    for split in ("train", "val"):
        path = Path(str(directories[split]))
        if not path.is_dir():
            raise FileNotFoundError(f"configured BDD100K {split} label directory is missing: {path}")
        count = sum(1 for _ in path.glob("*.json"))
        if count <= 0:
            raise ValueError(f"configured BDD100K {split} label directory has no direct JSON files: {path}")
        resolved[split] = str(path)
    return {"passed": True, "label_directories": resolved, "parser": "explicit_per_frame_json"}


def _run(command: Sequence[str], *, root: Path) -> dict[str, Any]:
    result = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {
        "passed": result.returncode == 0,
        "command": list(command),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def validate_smoke_artifacts(
    output_dir: str | Path,
    *,
    expected_git_head: str | None = None,
    expected_config_sha256: str | None = None,
    expected_source_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    required = ("run_manifest.json", "config_resolved.yaml", "source_fingerprint.json", "mechanism_stats.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"minimal real smoke artifacts missing: {missing}")
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "smoke" or manifest.get("synthetic") is not False:
        raise ValueError("smoke manifest must identify a real non-synthetic smoke")
    if expected_git_head is not None and manifest.get("git_head") != expected_git_head:
        raise ValueError("minimal real smoke is stale for the current git HEAD")
    if expected_config_sha256 is not None and manifest.get("config_sha256") != expected_config_sha256:
        raise ValueError("minimal real smoke is stale for the resolved config")
    source_fingerprint = json.loads((root / "source_fingerprint.json").read_text(encoding="utf-8"))
    if expected_source_fingerprint is not None and source_fingerprint != dict(expected_source_fingerprint):
        raise ValueError("minimal real smoke source fingerprint does not match current source")
    rows = [json.loads(line) for line in (root / "mechanism_stats.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("mechanism_stats.jsonl is empty")
    required_row = ("dino_call_count", "optimizer_step", "finite", "owner_parameter_delta")
    for row in rows:
        if row.get("placeholder") is True or any(name not in row for name in required_row):
            raise ValueError("mechanism artifact is placeholder or incomplete")
        if row["dino_call_count"] != 1 or row["finite"] is not True:
            raise ValueError("smoke artifact violates one-DINO/finite contract")
        if not isinstance(row["owner_parameter_delta"], Mapping) or not any(float(value) > 0.0 for value in row["owner_parameter_delta"].values()):
            raise ValueError("smoke artifact does not prove a real owner update")
    return {"passed": True, "row_count": len(rows), "root": str(root)}


def build_audit_record(
    *,
    git_head: str,
    required_files: Mapping[str, bool],
    required_tests: Mapping[str, bool],
    forbidden_results: Mapping[str, Sequence[str]],
    compile_result: Mapping[str, Any],
    pytest_result: Mapping[str, Any],
    smoke_result: Mapping[str, Any],
    pilot_override: Mapping[str, Any],
    grounding_layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not _SHA.fullmatch(git_head):
        raise ValueError("audit git_head must be a full SHA")
    missing = sorted([name for name, present in {**required_files, **required_tests}.items() if not present])
    forbidden = {name: list(paths) for name, paths in forbidden_results.items() if paths}
    grounding_ok = isinstance(grounding_layout, Mapping) and grounding_layout.get("passed") is True
    override_ok = (
        pilot_override.get("pilot_protocol_override") is True
        and pilot_override.get("pilot_completed") is False
        and pilot_override.get("replacement") == "minimal_real_smoke_only"
        and bool(pilot_override.get("reason"))
    )
    passed = (
        not missing
        and not forbidden
        and grounding_ok
        and override_ok
        and compile_result.get("passed") is True
        and pytest_result.get("passed") is True
        and smoke_result.get("passed") is True
    )
    unresolved = [
        *missing,
        *[f"forbidden:{name}" for name in forbidden],
        *([] if grounding_ok else ["grounding_layout"]),
        *([] if override_ok else ["explicit_pilot_override"]),
        *([] if compile_result.get("passed") is True else ["py_compile"]),
        *([] if pytest_result.get("passed") is True else ["targeted_pytest"]),
        *([] if smoke_result.get("passed") is True else ["minimal_real_smoke"]),
    ]
    return {
        "schema_version": "rael-p21-audit-v1",
        "pass": passed,
        "git_head": git_head,
        "required_files": dict(required_files),
        "required_tests": dict(required_tests),
        "forbidden_pattern_results": {name: list(paths) for name, paths in forbidden_results.items()},
        "compile_result": dict(compile_result),
        "pytest_result": dict(pytest_result),
        "smoke_result": dict(smoke_result),
        "grounding_layout": None if grounding_layout is None else dict(grounding_layout),
        "pilot_override": dict(pilot_override),
        "missing_items": missing,
        "unresolved": unresolved,
        "warnings": ["Original 3-epoch pilot was explicitly user-overridden; no pilot completion is claimed."],
    }


def write_review_pass(output_dir: str | Path, record: Mapping[str, Any]) -> Path:
    if record.get("pass") is not True:
        raise ValueError("refusing to write REVIEW_PASS for a failed RAEL audit")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "REVIEW_PASS_RAEL_OIA_V1.txt"
    path.write_text(f"pass=true\ngit_head={record['git_head']}\n", encoding="utf-8")
    return path


def write_full_train_gate(repository_root: str | Path, record: Mapping[str, Any]) -> Path:
    if record.get("pass") is not True or record.get("unresolved") != []:
        raise ValueError("refusing to write FULL_TRAIN_READY for an unresolved RAEL audit")
    override = record.get("pilot_override")
    smoke = record.get("smoke_result")
    if not isinstance(override, Mapping) or override.get("replacement") != "minimal_real_smoke_only":
        raise ValueError("FULL_TRAIN_READY requires the explicit user pilot replacement")
    if not isinstance(smoke, Mapping) or smoke.get("passed") is not True:
        raise ValueError("FULL_TRAIN_READY requires a passing minimal real smoke")
    destination = Path(repository_root) / ".review"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "RAEL_OIA_V1_FULL_TRAIN_READY.json"
    payload = {
        "schema_version": "rael-full-train-ready-v1",
        "pass": True,
        "git_head": record["git_head"],
        "pilot_override": dict(override),
        "smoke_result": dict(smoke),
        "unresolved": [],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_audit(
    *,
    repository_root: str | Path,
    output_dir: str | Path,
    config: str | Path,
    smoke_dir: str | Path | None,
    pilot_override_reason: str | None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_path = root / config
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(resolved_config, Mapping):
        raise ValueError("RAEL config must be a mapping")
    try:
        grounding_layout = validate_explicit_grounding_layout(resolved_config)
    except Exception as error:
        grounding_layout = {"passed": False, "reason": str(error)}
    required_files = _required_files(root, REQUIRED_SOURCE_FILES)
    required_tests = _required_files(root, REQUIRED_TEST_FILES)
    compile_targets = [str(root / name) for name, present in required_files.items() if present and name.endswith(".py")]
    compile_result = _run([sys.executable, "-m", "py_compile", *compile_targets], root=root) if compile_targets else {"passed": False, "command": []}
    pytest_result = _run([sys.executable, "-m", "pytest", "-q", *[name for name, present in required_tests.items() if present]], root=root) if all(required_tests.values()) else {"passed": False, "command": []}
    git_head = _git_head(root)
    config_sha256 = hashlib.sha256(
        json.dumps(resolved_config, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        from fate_oia.engine.train_acpr_rael_oia import build_rael_repository_fingerprints

        expected_source_fingerprint = build_rael_repository_fingerprints(root)
    except Exception as error:
        expected_source_fingerprint = {"audit_source_fingerprint_error": str(error)}
    try:
        smoke_result = (
            validate_smoke_artifacts(
                smoke_dir,
                expected_git_head=git_head,
                expected_config_sha256=config_sha256,
                expected_source_fingerprint=expected_source_fingerprint,
            )
            if smoke_dir is not None
            else {"passed": False, "reason": "minimal real smoke was not supplied"}
        )
    except Exception as error:
        smoke_result = {"passed": False, "reason": str(error)}
    if pilot_override_reason is None:
        override = {
            "pilot_protocol_override": False,
            "pilot_completed": False,
            "reason": None,
            "replacement": None,
        }
    else:
        override = pilot_override_payload(enabled=True, reason=pilot_override_reason)
    record = build_audit_record(
        git_head=git_head,
        required_files=required_files,
        required_tests=required_tests,
        forbidden_results=_forbidden_results(root),
        compile_result=compile_result,
        pytest_result=pytest_result,
        smoke_result=smoke_result,
        pilot_override=override,
        grounding_layout=grounding_layout,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if record["pass"]:
        record["review_pass_path"] = str(write_review_pass(destination, record))
        record["full_train_gate_path"] = str(write_full_train_gate(root, record))
    (destination / "implementation_audit_RAEL_OIA_V1.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed RAEL P21 audit")
    parser.add_argument("--mode", choices=("preflight",), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--smoke_dir")
    parser.add_argument("--pilot-override-reason")
    parser.add_argument("--repository_root", default=".")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    result = run_audit(repository_root=args.repository_root, output_dir=args.output_dir, config=args.config, smoke_dir=args.smoke_dir, pilot_override_reason=args.pilot_override_reason)
    print(json.dumps({"pass": result["pass"], "git_head": result["git_head"], "missing_items": result["missing_items"]}, sort_keys=True))
    return 0 if result["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
