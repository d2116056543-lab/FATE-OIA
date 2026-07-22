from __future__ import annotations

import argparse
import compileall
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from fate_oia.models.precise_oia_model import PRECISEOIAModel


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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _tree_sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path).encode("utf-8")); digest.update(path.read_bytes())
    return digest.hexdigest()


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
    scan_paths = [root / item for item in REQUIRED if Path(item).suffix.lower() in {".py", ".yaml", ".ps1"}]
    # The audit file necessarily contains the forbidden literals as rules.
    scan_paths = [path for path in scan_paths if path.name != "audit_precise_oia_implementation.py"]
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


def _pilot_checks(pilot_dir: Path) -> dict[str, bool]:
    mechanism = _jsonl(pilot_dir / "mechanism_batch_stats.jsonl")
    gradients = _jsonl(pilot_dir / "gradient_ownership.jsonl")
    metrics = _jsonl(pilot_dir / "metrics_summary.jsonl")
    pcvl_path = pilot_dir / "pcvl" / "pcvl_metrics.json"
    pcvl = json.loads(pcvl_path.read_text(encoding="utf-8")) if pcvl_path.exists() else {}
    latest_mechanism = mechanism[-1] if mechanism else {}
    latest_gradient = gradients[-1] if gradients else {}
    latest_metrics = metrics[-1] if metrics else {}
    owners = latest_gradient.get("owners", {})
    optimizer_counts = latest_metrics.get("optimizer_step_counts", {})
    firewall = {key: value for key, value in latest_gradient.items() if key.startswith("observed_to_") and key != "observed_to_annotation_adapter_grad_norm"}
    branch = latest_metrics
    reason_semantic = branch.get("reason_semantic", {})
    evidence_shuffled = branch.get("reason_evidence_shuffled", {})
    return {
        "three_epochs_complete": len(metrics) == 3,
        "mechanism_rows_present": bool(mechanism),
        "all_values_finite": _finite(mechanism) and _finite(gradients) and _finite(metrics) and _finite(pcvl),
        "all_intended_owners_stepped": bool(owners) and bool(optimizer_counts) and all(float(value) > 0 for value in optimizer_counts.values()),
        "observed_firewall_exact_zero": bool(firewall) and all(float(value) == 0.0 for value in firewall.values()),
        "action_exchange_ratio_in_range": 0.01 <= float(latest_mechanism.get("action_exchange_to_direct_ratio", -1)) <= 0.25,
        "reason_exchange_ratio_in_range": 0.01 <= float(latest_mechanism.get("reason_exchange_to_direct_ratio", -1)) <= 0.30,
        "reliability_noncollapsed": 0.0 < float(latest_mechanism.get("explicit_reliability_mean", -1)) < 1.0,
        "reference_not_center_collapsed": float(latest_mechanism.get("reference_center_collapse_rate", 1.0)) < 0.70,
        "selected_beats_control": float(latest_mechanism.get("selected_effect_mean", -1)) > float(latest_mechanism.get("control_effect_mean", 0)),
        "evidence_shuffle_changes_reason": bool(reason_semantic) and bool(evidence_shuffled) and abs(float(reason_semantic.get("Exp_mAP", 0)) - float(evidence_shuffled.get("Exp_mAP", 0))) > 1e-6,
        "annotation_delta_nonzero": any((pilot_dir / f"epoch_{epoch:03d}" / "annotation_gap.json").exists() and json.loads((pilot_dir / f"epoch_{epoch:03d}" / "annotation_gap.json").read_text(encoding="utf-8")).get("annotation_delta_rms", 0) > 0 for epoch in range(3)),
        "pcvl_artifacts_complete": all((pilot_dir / "pcvl" / name).exists() for name in ("pcvl_metrics.json", "pcvl_per_action.json", "pcvl_bootstrap.json", "pcvl_value_decomposition.json")),
        "pcvl_predicate_action_value_supported": bool(pcvl.get("predicate_action_value_supported")) and float(pcvl.get("u1_action_map", 0)) > float(pcvl.get("u0_action_map", 0)),
    }


def run_audit(config_path: str | Path, output_dir: str | Path, mode: str, pilot_dir: str | Path | None = None, write_gate: bool = False, device_name: str = "cuda") -> dict:
    root = Path.cwd()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    missing = [path for path in REQUIRED if not (root / path).exists()]
    source = list((root / "fate_oia").rglob("precise_*.py"))
    review_sources = [root / item for item in REQUIRED if (root / item).exists()]
    compile_ok = compileall.compile_dir(root / "fate_oia", quiet=1)
    test_files = sorted(str(path) for path in (root / "tests").glob("test_precise_*.py"))
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
    checks = {
        "mock_forward_contract": list(forward["action_logits_final_raw"].shape) == [1, 4] and list(forward["reason_logits_final_raw"].shape) == [1, 21],
        "dino_call_one": int(forward["diagnostics"]["dino_call_count"]) == 1,
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
    }
    clean_tree = not _git(["status", "--porcelain"])
    preflight_pass = not missing and compile_ok and not clean_hits and not incomplete_hits and clean_tree and all(checks.values())
    pilot_checks = _pilot_checks(Path(pilot_dir)) if mode == "pilot" and pilot_dir else {}
    if mode == "pilot":
        status = "FULL_TRAIN_READY" if preflight_pass and pilot_checks and all(pilot_checks.values()) else "SCIENTIFIC_GATE_FAILED"
    else:
        status = "PRE_PILOT_ELIGIBLE" if preflight_pass else "CHANGES_REQUIRED"
    unresolved = missing + list(clean_hits) + list(incomplete_hits) + [name for name, passed in checks.items() if not passed]
    if not clean_tree:
        unresolved.append("git_worktree_dirty")
    if pilot_checks:
        unresolved.extend(name for name, passed in pilot_checks.items() if not passed)
    record = {"status": status, "git_head": head, "branch": _git(["branch", "--show-current"]), "base_commit": "373aa49feac17372574fd7fb056c1d79c7c848fe", "config_sha256": config_sha, "skill_sha256": _sha(repo_skill), "user_skill_path": str(user_skill), "source_tree_sha256": _tree_sha(review_sources), "checked_files": list(REQUIRED), "missing": missing, "forbidden_pattern_results": clean_hits, "incomplete_implementation_results": incomplete_hits, "functional_checks": checks, "pilot_checks": pilot_checks, "compile_ok": compile_ok, "pytest": {"passed": pytest_ok, "returncode": pytest_run.returncode, "stdout_tail": pytest_run.stdout[-4000:], "stderr_tail": pytest_run.stderr[-4000:]}, "git_clean": clean_tree, "mode": mode, "unresolved": sorted(set(unresolved))}
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
