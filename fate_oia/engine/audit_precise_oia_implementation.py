from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from fate_oia.models.precise_oia_model import PRECISEOIAModel
from fate_oia.engine.eval_precise_oia import EVAL_BRANCHES
from fate_oia.engine.run_precise_pcvl import validate_pcvl_artifacts
from fate_oia.metrics import multilabel_metrics_from_logits


REQUIRED = (
    "configs/fate_oia_train_360x640_precise_oia_v1.yaml", "configs/precise_evidence_fields.yaml", "configs/precise_reason_semantics.yaml", "configs/precise_action_semantics.yaml",
    "fate_oia/datasets/bdd100k_task_aware_index.py", "fate_oia/datasets/precise_grounding_adapter.py", "fate_oia/transforms_precise.py",
    "fate_oia/models/precise_dino_field.py", "fate_oia/models/precise_visual_field.py", "fate_oia/models/precise_category_decoder.py", "fate_oia/models/precise_evidence_fields.py", "fate_oia/models/precise_visual_rereader.py", "fate_oia/models/precise_semantic_exchange.py", "fate_oia/models/precise_annotation_head.py", "fate_oia/models/precise_pcvl_probes.py", "fate_oia/models/precise_oia_model.py",
    "fate_oia/losses/precise_losses.py", "fate_oia/losses/precise_intervention_losses.py", "fate_oia/utils/precise_schema.py", "fate_oia/utils/precise_artifacts.py", "fate_oia/utils/precise_gradient_ownership.py", "fate_oia/utils/precise_runtime.py",
    "fate_oia/engine/train_precise_oia.py", "fate_oia/engine/eval_precise_oia.py", "fate_oia/engine/run_precise_pcvl.py", "fate_oia/engine/profile_precise_oia.py", "fate_oia/engine/audit_precise_oia_implementation.py", "fate_oia/engine/export_precise_cases.py", "fate_oia/engine/supervise_precise_oia_foreground.py", "scripts/FATE_OIA_precise_oia_v1_foreground.ps1",
)

FORBIDDEN = (
    "ACPROIAModel", "ACPRLabelTrunk", "ACPRScenePredicateHead", "WeakPredicateTargetBuilder",
    "ACPRPredicateReasoner", "ACPRPairMemory", "ACPRActionComboAux", "ACPRCalibrationHead",
    "reason_gt_to_action", "reason_logits_observed_to_action", "Start-Process", "Start-Job", "nohup",
    "Register-ScheduledTask", "feature_cache_enabled: true", "token_compression: keep_merge",
)

REQUIRED_PILOT_CHECKS = {
    "pilot_identity_matches_current_code", "pilot_sample_contract", "three_epochs_complete",
    "mechanism_rows_present", "mechanism_epoch_coverage", "epoch_artifacts_complete", "all_values_finite", "dino_call_count_one",
    "peak_reserved_under_hard_limit", "all_intended_owners_stepped", "pcvl_optimizer_stepped",
    "observed_firewall_exact_zero", "action_exchange_ratio_in_range", "reason_exchange_ratio_in_range",
    "action_reread_ratio_in_range", "reason_reread_ratio_in_range", "reliability_noncollapsed",
    "reference_not_center_collapsed", "selected_beats_control", "evidence_shuffle_changes_reason",
    "annotation_delta_nonzero", "pcvl_artifacts_complete", "pcvl_predicate_action_value_supported",
    "pcvl_learned_evidence_supported", "pcvl_learned_exchange_supported",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _tree_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


def _training_source_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.glob("fate_oia/**/*precise*.py")):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _index_sha(indices: list[int]) -> str:
    return hashlib.sha256(",".join(str(index) for index in indices).encode("ascii")).hexdigest()


def _expected_pilot_index_hashes(total: int, seed: int) -> dict[str, str]:
    order = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()[:5632]
    return {
        "train_main_indices_sha256": _index_sha(order[:4096]),
        "train_audit_indices_sha256": _index_sha(order[4096:5120]),
        "train_calib_indices_sha256": _index_sha(order[5120:5632]),
    }


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(value) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(torch.isfinite(torch.tensor(float(value))))
    return True


def _scan_forbidden(root: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    scan_paths = set(root.glob("fate_oia/**/*precise*.py"))
    scan_paths.update(root.glob("configs/*precise*.yaml"))
    scan_paths.update(root.glob("scripts/*precise*.ps1"))
    scan_paths.update(root / item for item in REQUIRED if Path(item).suffix.lower() in {".py", ".yaml", ".ps1"})
    # The audit file necessarily contains the forbidden literals as rules.
    scan_paths = [path for path in sorted(scan_paths) if path.name != "audit_precise_oia_implementation.py"]
    forbidden_hits = {token: [] for token in FORBIDDEN}
    incomplete_hits = {token: [] for token in ("TODO", "FIXME", "NotImplementedError", "placeholder output")}
    pass_pattern = re.compile(r"(?m)^\s*pass\s*(?:#.*)?$")
    for path in scan_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        relative = str(path.relative_to(root))
        for token in FORBIDDEN:
            if token in text:
                forbidden_hits[token].append(relative)
        for token in incomplete_hits:
            if token.lower() in text.lower():
                incomplete_hits[token].append(relative)
        if pass_pattern.search(text):
            incomplete_hits.setdefault("bare_pass", []).append(relative)
    return ({key: value for key, value in forbidden_hits.items() if value}, {key: value for key, value in incomplete_hits.items() if value})


def _pilot_checks(pilot_dir: Path, expected_identity: dict[str, str]) -> dict[str, bool]:
    mechanism = _jsonl(pilot_dir / "mechanism_batch_stats.jsonl")
    gradients = _jsonl(pilot_dir / "gradient_ownership.jsonl")
    metrics = _jsonl(pilot_dir / "metrics_summary.jsonl")
    pcvl_path = pilot_dir / "pcvl" / "pcvl_metrics.json"
    pcvl = json.loads(pcvl_path.read_text(encoding="utf-8")) if pcvl_path.exists() else {}
    bootstrap_path = pilot_dir / "pcvl" / "pcvl_bootstrap.json"
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8")) if bootstrap_path.exists() else {}
    manifest_path = pilot_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    latest_mechanism = mechanism[-1] if mechanism else {}
    latest_gradient = gradients[-1] if gradients else {}
    latest_metrics = metrics[-1] if metrics else {}
    owners = latest_gradient.get("owners", {})
    optimizer_counts = latest_metrics.get("optimizer_step_counts", {})
    firewall = {key: value for key, value in latest_gradient.items() if key.startswith("observed_to_") and key != "observed_to_annotation_adapter_grad_norm"}
    branch = latest_metrics
    reason_semantic = branch.get("reason_semantic", {})
    evidence_shuffled = branch.get("reason_evidence_shuffled", {})
    final_epoch = max((int(row.get("epoch", -1)) for row in mechanism), default=-1)
    final_rows = [row for row in mechanism if int(row.get("epoch", -1)) == final_epoch]
    def median(name: str, default: float = -1.0) -> float:
        values = [float(row[name]) for row in final_rows if name in row]
        return float(statistics.median(values)) if values else default
    heldout_counterfactual = latest_metrics.get("counterfactual", {})
    expected_indices = _expected_pilot_index_hashes(int(manifest.get("train_dataset_count", 0)), int(manifest.get("seed", -1))) if int(manifest.get("train_dataset_count", 0)) >= 5632 else {}
    representation_owners = {name: value for name, value in owners.items() if name != "threshold_head"}
    owner_names = set(representation_owners)
    owner_epoch_activity = bool(owner_names) and all(
        any(
            int(row.get("epoch", -1)) == epoch
            and all(float(row.get("owners", {}).get(owner, {}).get("parameter_delta_norm", 0.0)) > 0.0 for owner in owner_names)
            for row in gradients
        )
        for epoch in range(3)
    )
    epoch_required = {
        "metrics_summary.json", "branch_metrics.json", "per_action_metrics.json", "per_reason_metrics.json",
        "evidence_family_stats.json", "evidence_reliability.json", "exchange_stats.json", "reread_stats.json",
        "annotation_gap.json", "counterfactual_stats.json", "gradient_firewall.json", "failure_cases.jsonl",
        "evidence_cases.jsonl", "labels_action.pt", "labels_reason.pt", "file_names.json",
    }
    def complete_epoch(epoch: int) -> bool:
        directory = pilot_dir / f"epoch_{epoch:03d}"
        tensors = {f"logits_{name}.pt" for name in EVAL_BRANCHES}
        required = epoch_required | tensors
        if not directory.is_dir() or not all((directory / name).exists() for name in required):
            return False
        try:
            expected_samples = int(manifest["test_count"])
            action_labels = torch.load(directory / "labels_action.pt", map_location="cpu", weights_only=True)
            reason_labels = torch.load(directory / "labels_reason.pt", map_location="cpu", weights_only=True)
            names = json.loads((directory / "file_names.json").read_text(encoding="utf-8"))["file_names"]
            if action_labels.shape != (expected_samples, 4) or reason_labels.shape != (expected_samples, 21) or len(names) != expected_samples:
                return False
            if not torch.isfinite(action_labels).all() or not torch.isfinite(reason_labels).all():
                return False
            for branch in EVAL_BRANCHES:
                logits = torch.load(directory / f"logits_{branch}.pt", map_location="cpu", weights_only=True)
                width = 4 if branch.startswith("action_") else 21
                if logits.shape != (expected_samples, width) or not torch.isfinite(logits).all():
                    return False
                target = action_labels if branch.startswith("action_") else reason_labels
                prefix = "Act_" if branch.startswith("action_") else "Exp_"
                recomputed = multilabel_metrics_from_logits(logits, target, prefix=prefix)
                recorded = json.loads((directory / "metrics_summary.json").read_text(encoding="utf-8")).get(branch, {})
                for key in (f"{prefix}mF1", f"{prefix}oF1", f"{prefix}mAP"):
                    if key in recomputed and (key not in recorded or abs(float(recomputed[key]) - float(recorded[key])) > 1e-8):
                        return False
            return True
        except (OSError, ValueError, KeyError, RuntimeError, TypeError):
            return False
    pcvl_identity_keys = (
        "git_head", "source_tree_sha256", "config_sha256", "skill_sha256",
        "pretrained_weights_sha256", "action_schema_sha256",
        "train_audit_indices_sha256", "train_audit_file_names_sha256",
    )
    pcvl_expected = {key: manifest.get(key) for key in pcvl_identity_keys}
    try:
        validate_pcvl_artifacts(pilot_dir / "pcvl", expected_identity=pcvl_expected)
        pcvl_artifacts_valid = True
    except (OSError, ValueError, KeyError, RuntimeError, TypeError, json.JSONDecodeError):
        pcvl_artifacts_valid = False
    checks = {
        "pilot_identity_matches_current_code": all(manifest.get(key) == value for key, value in expected_identity.items()),
        "pilot_sample_contract": manifest.get("train_main_count") == 4096 and manifest.get("train_audit_count") == 1024 and manifest.get("train_calib_count") == 512 and manifest.get("test_count") == 512 and manifest.get("seed") == 20260722 and manifest.get("epochs") == 3 and bool(expected_indices) and all(manifest.get(key) == value for key, value in expected_indices.items()),
        "three_epochs_complete": sorted(int(row.get("epoch", -1)) for row in metrics) == [0, 1, 2],
        "mechanism_rows_present": bool(mechanism),
        "mechanism_epoch_coverage": all(any(int(row.get("epoch", -1)) == epoch for row in mechanism) for epoch in range(3)),
        "epoch_artifacts_complete": all(complete_epoch(epoch) for epoch in range(3)) and (pilot_dir / "checkpoint_latest.pth").exists(),
        "all_values_finite": _finite(mechanism) and _finite(gradients) and _finite(metrics) and _finite(pcvl),
        "dino_call_count_one": bool(mechanism) and all(int(row.get("dino_call_count_batch", -1)) == 1 for row in mechanism),
        "peak_reserved_under_hard_limit": bool(mechanism) and max(float(row.get("gpu_peak_reserved_gb", float("inf"))) for row in mechanism) < 46.5,
        "all_intended_owners_stepped": owner_epoch_activity and bool(optimizer_counts) and all(float(value) > 0 for value in optimizer_counts.values()),
        "pcvl_optimizer_stepped": int(latest_metrics.get("pcvl_optimizer_step_count", 0)) > 0 and int(latest_metrics.get("pcvl_nonzero_update_count", 0)) > 0 and median("pcvl_grad_norm", 0) > 0 and median("pcvl_parameter_delta_norm", 0) > 0,
        "observed_firewall_exact_zero": bool(firewall) and all(float(value) == 0.0 for value in firewall.values()),
        "action_exchange_ratio_in_range": 0.01 <= median("action_exchange_to_direct_ratio") <= 0.25,
        "reason_exchange_ratio_in_range": 0.01 <= median("reason_exchange_to_direct_ratio") <= 0.30,
        "action_reread_ratio_in_range": 0.02 <= median("action_reread_to_direct_ratio") <= 0.30,
        "reason_reread_ratio_in_range": 0.02 <= median("reason_reread_to_direct_ratio") <= 0.30,
        "reliability_noncollapsed": 0.0 < median("explicit_reliability_mean") < 1.0 and median("explicit_reliability_std", 0.0) > 1e-6,
        "reference_not_center_collapsed": median("reference_center_collapse_rate", 1.0) < 0.70,
        "selected_beats_control": float(heldout_counterfactual.get("selected_control_margin", -float("inf"))) > 0.0,
        "evidence_shuffle_changes_reason": bool(reason_semantic) and bool(evidence_shuffled) and abs(float(reason_semantic.get("Exp_mAP", 0)) - float(evidence_shuffled.get("Exp_mAP", 0))) >= 1e-4,
        "annotation_delta_nonzero": all((pilot_dir / f"epoch_{epoch:03d}" / "annotation_gap.json").exists() and json.loads((pilot_dir / f"epoch_{epoch:03d}" / "annotation_gap.json").read_text(encoding="utf-8")).get("annotation_delta_rms", 0) >= 1e-5 for epoch in range(3)),
        "pcvl_artifacts_complete": pcvl_artifacts_valid,
        "pcvl_predicate_action_value_supported": bool(pcvl.get("predicate_action_value_supported")) and float(pcvl.get("u1_action_map", 0)) > float(pcvl.get("u0_action_map", 0)) and float(bootstrap.get("delta_value", {}).get("ci_low", -1.0)) > 0.0 and float(bootstrap.get("delta_value", {}).get("positive_rate", 0.0)) >= 0.95,
        "pcvl_learned_evidence_supported": float(pcvl.get("u2_action_map", 0)) > float(pcvl.get("u0_action_map", 0)) and float(bootstrap.get("delta_learned_value", {}).get("ci_low", -1.0)) > 0.0,
        "pcvl_learned_exchange_supported": float(pcvl.get("u3_action_map", 0)) > float(pcvl.get("u2_action_map", 0)) and float(bootstrap.get("delta_learned_interaction", {}).get("ci_low", -1.0)) > 0.0,
    }
    if set(checks) != REQUIRED_PILOT_CHECKS:
        raise RuntimeError(f"Pilot check schema drift: {sorted(set(checks) ^ REQUIRED_PILOT_CHECKS)}")
    return checks


def run_audit(config_path: str | Path, output_dir: str | Path, mode: str, pilot_dir: str | Path | None = None, write_gate: bool = False, device_name: str = "cuda") -> dict:
    root = Path.cwd()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dino_path = root / config["pretrained_weights"]
    dino_sha = _sha(dino_path) if dino_path.is_file() else "missing"
    action_schema_path = root / "configs" / "precise_action_semantics.yaml"
    action_schema_sha = _sha(action_schema_path) if action_schema_path.is_file() else "missing"
    missing = [path for path in REQUIRED if not (root / path).exists()]
    source = list((root / "fate_oia").rglob("precise_*.py"))
    review_sources = [root / item for item in REQUIRED if (root / item).exists()]
    compile_ok = compileall.compile_dir(root / "fate_oia", quiet=1)
    test_files = sorted(str(path) for path in (root / "tests").glob("test_precise_*.py"))
    regression_names = (
        "test_acpr_calalign_forward.py", "test_acpr_calalign_train_protocol.py",
        "test_acpr_threshold_head.py", "test_acpr_threshold_losses.py", "test_acpr_threshold_search.py",
    )
    test_files.extend(str(root / "tests" / name) for name in regression_names if (root / "tests" / name).exists())
    pytest_run = subprocess.run([sys.executable, "-m", "pytest", *test_files, "-q"], capture_output=True, text=True, check=False)
    pytest_ok = pytest_run.returncode == 0
    clean_hits, incomplete_hits = _scan_forbidden(root)
    device = torch.device(device_name if device_name.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = PRECISEOIAModel(Path(config_path).parent, use_mock_dino=True, model_config=config).to(device)
    with torch.no_grad():
        forward = model(torch.randn(1, 3, 360, 640, device=device))
    runtime_file = root / ".review" / "precise_oia_v1" / "runtime" / "selected_runtime_profile.json"
    real_forward_file = root / ".review" / "precise_oia_v1" / "real_forward.json"
    runtime = json.loads(runtime_file.read_text(encoding="utf-8")) if runtime_file.exists() else {}
    real_forward = json.loads(real_forward_file.read_text(encoding="utf-8")) if real_forward_file.exists() else {}
    trainer_source = (root / "fate_oia/engine/train_precise_oia.py").read_text(encoding="utf-8")
    supervisor_source = (root / "fate_oia/engine/supervise_precise_oia_foreground.py").read_text(encoding="utf-8")
    profiler_source = (root / "fate_oia/engine/profile_precise_oia.py").read_text(encoding="utf-8")
    pcvl_source = (root / "fate_oia/engine/run_precise_pcvl.py").read_text(encoding="utf-8")
    repo_skill = root / ".codex/skills/precise-oia-implementation-audit/SKILL.md"
    user_skill = Path.home() / ".codex/skills/precise-oia-implementation-audit/SKILL.md"
    head = _git(["rev-parse", "HEAD"])
    config_sha = _sha(Path(config_path))
    source_tree_sha = _tree_sha(review_sources)
    branch = _git(["branch", "--show-current"])
    base_commit = "373aa49feac17372574fd7fb056c1d79c7c848fe"
    base_is_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", base_commit, "HEAD"], check=False).returncode == 0
    checks = {
        "mock_forward_contract": list(forward["action_logits_final_raw"].shape) == [1, 4] and list(forward["reason_logits_final_raw"].shape) == [1, 21],
        "dino_call_one": int(forward["diagnostics"]["dino_call_count"]) == 1,
        "official_dino_exact_identity": dino_sha != "missing" and bool(real_forward.get("dino_loaded_state_key_count", 0)) and not real_forward.get("dino_missing_keys") and not real_forward.get("dino_unexpected_keys") and real_forward.get("dino_weights_sha256") == dino_sha,
        "action_semantics_runtime_wired": [row["name"] for row in model.action_schema] == ["forward", "stop", "left", "right"] and "load_action_semantics" in (root / "fate_oia/models/precise_oia_model.py").read_text(encoding="utf-8"),
        "test_only": config["eval_splits"] == "test" and config["best_selection_split"] == "test",
        "no_compression": config["token_compression"] == "none" and not config["feature_cache_enabled"],
        "annotation_firewall_source": "semantic_logits.detach() + delta" in (root / "fate_oia/models/precise_annotation_head.py").read_text(encoding="utf-8"),
        "actual_runtime_profile": bool(runtime.get("valid")) and int(runtime.get("dino_call_count", 0)) == 1 and runtime.get("git_head") == head and runtime.get("config_sha256") == config_sha,
        "real_forward_passed": bool(real_forward.get("passed")) and real_forward.get("git_head") == head and real_forward.get("config_sha256") == config_sha,
        "gradient_firewall_passed": bool(real_forward.get("gradient_firewall_passed")),
        "owner_gradient_matrix_passed": bool(real_forward.get("owner_gradient_matrix_passed")),
        "curve_distance_supervision_active": float(real_forward.get("curve_distance_valid_count", 0.0)) > 0.0,
        # A module that is merely importable is not an implementation.  These
        # checks deliberately inspect the formal training/pilot call chain.
        "intervention_called_by_trainer": "packed_target_specific_interventions" in trainer_source,
        "mirror_consistency_called_by_trainer": "two_way_consistency_loss" in trainer_source,
        "refinement_called_by_trainer": "refinement_loss" in trainer_source,
        "train_calib_threshold_called": "make_train_calib_indices" in trainer_source and "update_teacher" in trainer_source,
        "yaml_model_config_wired": "model_config=config" in trainer_source and "model_config=config" in profiler_source and "model_config=config" in Path(__file__).read_text(encoding="utf-8"),
        "structured_pcvl_oracle": all(token in pcvl_source for token in ("part_coordinates", "part_scales", "soft_masks", "action_evidence_family_mask", "oracle_by_action")),
        "pcvl_called_by_pilot_supervisor": "run_precise_pcvl" in supervisor_source,
        "resume_scheduler_contract": "load_resume_checkpoint(" in trainer_source and all(name in trainer_source for name in ("optimizer.load_state_dict", "scheduler.load_state_dict", "torch.set_rng_state", "random.setstate", "torch.cuda.set_rng_state_all")),
        "multipart_evidence_called": "explicit_part_attention" in (root / "fate_oia/models/precise_evidence_fields.py").read_text(encoding="utf-8"),
        "latent_reason_path_called": "reason_latent_delta" in (root / "fate_oia/models/precise_oia_model.py").read_text(encoding="utf-8"),
        "active_schema_precedes_model": trainer_source.index("grounding_adapter, train_grounding, active_fields = build_train_grounding_targets") < trainer_source.index("PRECISEOIAModel(Path"),
        "all_diagnostic_branches": all(name in (root / "fate_oia/engine/eval_precise_oia.py").read_text(encoding="utf-8") for name in ("explicit_only", "latent_only", "exchange_off", "evidence_shuffled", "reason_token_shuffled", "annotation_off")),
        "bf16_and_owner_clip_called": "torch.autocast(" in trainer_source and "clip_grad_norm_(" in trainer_source,
        "targeted_pytest_passed": pytest_ok,
        "skill_installed_and_identical": user_skill.exists() and _sha(user_skill) == _sha(repo_skill),
        "branch_and_base_contract": branch == "acpr_precise_oia_v1_direct_image" and base_is_ancestor,
    }
    clean_tree = not _git(["status", "--porcelain"])
    preflight_pass = not missing and compile_ok and not clean_hits and not incomplete_hits and clean_tree and all(checks.values())
    expected_identity = {"git_head": head, "config_sha256": config_sha, "source_tree_sha256": _training_source_sha(root), "skill_sha256": _sha(repo_skill), "pretrained_weights_sha256": dino_sha, "action_schema_sha256": action_schema_sha}
    pilot_checks = _pilot_checks(Path(pilot_dir), expected_identity) if mode == "pilot" and pilot_dir else {}
    if mode == "pilot":
        status = "FULL_TRAIN_READY" if preflight_pass and pilot_checks and all(pilot_checks.values()) else "SCIENTIFIC_GATE_FAILED"
    else:
        status = "PRE_PILOT_ELIGIBLE" if preflight_pass else "CHANGES_REQUIRED"
    unresolved = missing + list(clean_hits) + list(incomplete_hits) + [name for name, passed in checks.items() if not passed]
    if not clean_tree:
        unresolved.append("git_worktree_dirty")
    if pilot_checks:
        unresolved.extend(name for name, passed in pilot_checks.items() if not passed)
    record = {"status": status, "git_head": head, "branch": branch, "base_commit": base_commit, "config_sha256": config_sha, "skill_sha256": _sha(repo_skill), "action_schema_sha256": action_schema_sha, "pretrained_weights_sha256": dino_sha, "user_skill_path": str(user_skill), "source_tree_sha256": source_tree_sha, "checked_files": list(REQUIRED), "missing": missing, "forbidden_pattern_results": clean_hits, "incomplete_implementation_results": incomplete_hits, "functional_checks": checks, "pilot_checks": pilot_checks, "compile_ok": compile_ok, "tests_passed": pytest_ok, "real_forward_passed": bool(real_forward.get("passed")), "gradient_firewall_passed": bool(real_forward.get("gradient_firewall_passed")), "dino_call_contract_passed": int(runtime.get("dino_call_count", 0)) == 1, "runtime_profile_passed": bool(runtime.get("valid")), "pytest": {"passed": pytest_ok, "returncode": pytest_run.returncode, "stdout_tail": pytest_run.stdout[-4000:], "stderr_tail": pytest_run.stderr[-4000:]}, "git_clean": clean_tree, "mode": mode, "unresolved": sorted(set(unresolved))}
    if mode == "pilot":
        record.update({"pilot_complete": pilot_checks.get("three_epochs_complete", False), "mechanisms_active": all(pilot_checks.get(name, False) for name in REQUIRED_PILOT_CHECKS if name not in {"pcvl_predicate_action_value_supported"}), "pcvl": json.loads((Path(pilot_dir) / "pcvl" / "pcvl_metrics.json").read_text(encoding="utf-8")) if (Path(pilot_dir) / "pcvl" / "pcvl_metrics.json").exists() else {}, "runtime_selected": runtime})
    (output / "implementation_audit_PRECISE_OIA_V1.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    gate = root / ".review" / "PRECISE_OIA_V1_PRE_PILOT_ELIGIBLE.json"
    gate.parent.mkdir(parents=True, exist_ok=True)
    if write_gate and status == "PRE_PILOT_ELIGIBLE":
        gate.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if write_gate and status == "FULL_TRAIN_READY":
        (root / ".review" / "PRECISE_OIA_V1_FULL_TRAIN_READY.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="preflight")
    parser.add_argument("--write_pre_pilot_eligible", action="store_true")
    parser.add_argument("--write_full_train_ready", action="store_true")
    parser.add_argument("--pilot_dir")
    args = parser.parse_args()
    write_gate = args.write_full_train_ready if args.mode == "pilot" else args.write_pre_pilot_eligible
    record = run_audit(args.config, args.output_dir, args.mode, args.pilot_dir, write_gate=write_gate, device_name=args.device)
    print(json.dumps(record, sort_keys=True))
    expected = "FULL_TRAIN_READY" if args.mode == "pilot" else "PRE_PILOT_ELIGIBLE"
    if record["status"] != expected:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
