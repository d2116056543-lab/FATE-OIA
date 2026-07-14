from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


class ICDORAuditError(RuntimeError):
    """Raised when IC-DOR evidence is incomplete or fails a hard protocol gate."""


_REQUIRED_FORWARD_OUTPUTS = (
    "action_final_logits",
    "reason_observed_logits",
    "factor_soft_masks",
    "support_weights",
    "veto_weights",
    "action_factor_off_logits",
    "action_factor_shuffled_logits",
    "action_wrong_target_logits",
    "action_equal_mass_random_logits",
    "reason_propensity",
)

_REQUIRED_FUNCTIONAL_CHECKS = (
    "direct_image",
    "factor_certificate",
    "edge_admission",
    "action_firewall",
    "reason_firewall",
    "selective_observation",
    "calibration",
    "artifact_schema",
    "resume_integrity",
    "visual_audit",
    "foreground_launcher",
)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _require_tensor(output: Mapping[str, Any], key: str, batch_size: int) -> torch.Tensor:
    value = output.get(key)
    if not isinstance(value, torch.Tensor):
        raise ICDORAuditError(f"dynamic forward did not produce tensor {key}")
    if value.ndim == 0 or value.shape[0] != batch_size:
        raise ICDORAuditError(f"dynamic forward produced invalid batch shape for {key}")
    if not torch.isfinite(value).all():
        raise ICDORAuditError(f"dynamic forward produced non-finite {key}")
    return value


def _gradient_sum_abs(model: nn.Module, output: torch.Tensor, parameter_prefix: str) -> float:
    model.zero_grad(set_to_none=True)
    output.float().sum().backward(retain_graph=True)
    total = 0.0
    matched = 0
    for name, parameter in model.named_parameters():
        if name.startswith(parameter_prefix):
            matched += 1
            if parameter.grad is not None:
                total += float(parameter.grad.detach().abs().sum().cpu())
    if matched == 0:
        raise ICDORAuditError(f"gradient audit found no parameters for {parameter_prefix}")
    return total


def _cross_gradient_sum_abs(model: nn.Module, output: torch.Tensor, parameter_prefix: str) -> float:
    """Return zero for an intentionally disconnected lane, but require the lane to exist."""
    model.zero_grad(set_to_none=True)
    output.float().sum().backward(retain_graph=True)
    matched = 0
    total = 0.0
    for name, parameter in model.named_parameters():
        if name.startswith(parameter_prefix):
            matched += 1
            if parameter.grad is not None:
                total += float(parameter.grad.detach().abs().sum().cpu())
    if matched == 0:
        raise ICDORAuditError(f"gradient firewall found no parameters for {parameter_prefix}")
    return total


def _lane_parameters(model: nn.Module, prefix: str) -> list[nn.Parameter]:
    parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
    if not parameters:
        raise ICDORAuditError(f"gradient audit found no parameters for {prefix}")
    return parameters


def _joint_lane_gradients(
    output: torch.Tensor,
    owned: list[nn.Parameter],
    crossed: list[nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[float, float]:
    gradients = torch.autograd.grad(
        output.float().sum(), [*owned, *crossed], retain_graph=retain_graph,
        allow_unused=True,
    )
    owned_count = len(owned)
    owned_sum = sum(float(grad.detach().abs().sum().cpu()) for grad in gradients[:owned_count] if grad is not None)
    crossed_sum = sum(float(grad.detach().abs().sum().cpu()) for grad in gradients[owned_count:] if grad is not None)
    return owned_sum, crossed_sum


def verify_dynamic_forward_and_gradients(model: nn.Module, images: torch.Tensor) -> dict[str, Any]:
    """Prove outputs, input sensitivity, and lane gradients through actual forwards."""
    if images.ndim != 4 or images.shape[0] < 1:
        raise ICDORAuditError("dynamic audit requires an image batch [B,C,H,W]")
    was_training = model.training
    model.train()
    try:
        first = model(
            images, route_mode="admitted", latent_enabled=True, reason_route_mode="full",
            return_masks=True, return_diagnostics=True,
        )
        second = model(
            images + torch.full_like(images, 0.137),
            route_mode="admitted",
            latent_enabled=True,
            reason_route_mode="full",
            return_masks=True,
        )
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise ICDORAuditError("dynamic forward must return a mapping")
        first_tensors = {key: _require_tensor(first, key, images.shape[0]) for key in _REQUIRED_FORWARD_OUTPUTS}
        second_tensors = {
            key: _require_tensor(second, key, images.shape[0])
            for key in ("action_final_logits", "reason_observed_logits", "factor_soft_masks")
        }
        sensitivity = {
            key: float((first_tensors[key].detach() - second_tensors[key].detach()).abs().max().cpu())
            for key in ("action_final_logits", "reason_observed_logits", "factor_soft_masks")
        }
        if any(value <= 0.0 for value in sensitivity.values()):
            raise ICDORAuditError(f"dynamic forward is input-insensitive: {sensitivity}")
        action_parameters = _lane_parameters(model, "action_adapter.")
        reason_parameters = _lane_parameters(model, "reason_adapter.")
        action_owned, action_cross = _joint_lane_gradients(
            first_tensors["action_final_logits"], action_parameters, reason_parameters, retain_graph=True
        )
        reason_owned, reason_cross = _joint_lane_gradients(
            first_tensors["reason_observed_logits"], reason_parameters, action_parameters, retain_graph=False
        )
        gradients = {"action_adapter": action_owned, "reason_adapter": reason_owned}
        if any(value < 1e-8 for value in gradients.values()):
            raise ICDORAuditError(f"dynamic gradient path is inactive: {gradients}")
        firewall = {"reason_to_action_adapter": reason_cross, "action_to_reason_adapter": action_cross}
        if any(value != 0.0 for value in firewall.values()):
            raise ICDORAuditError(f"dynamic gradient firewall is violated: {firewall}")
        return {
            "pass": True,
            "forward_calls": 2,
            "input_sensitivity": sensitivity,
            "gradient_sum_abs": gradients,
            "gradient_firewall": firewall,
            "required_outputs": list(_REQUIRED_FORWARD_OUTPUTS),
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def protocol_hard_gate(
    review_path: str | Path,
    *,
    target_head: str,
    config_sha256: str,
    runtime_sha256: str,
    required_gates: Iterable[str],
) -> dict[str, Any]:
    """Reject stale, incomplete, or hash-drifted REVIEW_PASS documents."""
    path = Path(review_path)
    if not path.is_file():
        raise ICDORAuditError("REVIEW_PASS is missing")
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ICDORAuditError(f"REVIEW_PASS is invalid JSON: {error}") from error
    if not isinstance(review, Mapping) or review.get("status") != "PASS":
        raise ICDORAuditError("REVIEW_PASS status is not PASS")
    if review.get("target_head") != target_head:
        raise ICDORAuditError("REVIEW_PASS target HEAD does not match current HEAD")
    if review.get("resolved_config_sha256") != config_sha256:
        raise ICDORAuditError("REVIEW_PASS config hash does not match requested config hash")
    if review.get("runtime_selection_sha256") != runtime_sha256:
        raise ICDORAuditError("REVIEW_PASS runtime hash does not match requested runtime hash")
    gates = review.get("gates")
    if not isinstance(gates, Mapping):
        raise ICDORAuditError("REVIEW_PASS gates are missing")
    missing = [gate for gate in required_gates if gates.get(gate) != "PASS"]
    if missing:
        raise ICDORAuditError(f"REVIEW_PASS has non-passing gates: {missing}")
    return dict(review)


def source_hard_gate(worktree_root: str | Path) -> dict[str, Any]:
    """Check the formal model source for direct-image and non-legacy call sites."""
    root = Path(worktree_root)
    source_path = root / "fate_oia" / "models" / "acpr_mosaic_trust_icdor_model.py"
    if not source_path.is_file():
        raise ICDORAuditError(f"formal model source is missing: {source_path}")
    source = source_path.read_text(encoding="utf-8")
    required = ("self.dino(images)", "self.action_adapter", "self.reason_adapter", "return_masks")
    missing = [token for token in required if token not in source]
    forbidden = ("MOSAICSupportVetoComposer(", "MOSAICActionDecoder(", "MOSAICReasonDecoder(")
    present = [token for token in forbidden if token in source]
    if missing or present:
        raise ICDORAuditError(f"source hard gate failed; missing={missing}, forbidden={present}")
    return {"pass": True, "model_sha256": sha256_file(source_path), "source": str(source_path)}


def verify_source_manifest(root: Path) -> dict[str, Any]:
    path = root / ".review" / "icdor_source_manifest.json"
    if not path.is_file():
        raise ICDORAuditError("IC-DOR source manifest is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "source_branch", "source_head", "target_branch", "target_worktree",
        "source_worktree_clean", "created_at", "plan_sha256", "audit_skill_sha256",
    }
    missing = sorted(required - set(payload))
    if missing or payload.get("source_worktree_clean") is not True:
        raise ICDORAuditError(f"IC-DOR source manifest is incomplete: {missing}")
    skill = root / ".codex" / "skills" / "acpr-mosaic-trust-v3-icdor-implementation-audit" / "SKILL.md"
    if not skill.is_file() or sha256_file(skill) != payload["audit_skill_sha256"]:
        raise ICDORAuditError("IC-DOR audit skill hash drifted from source manifest")
    if payload.get("target_branch") != "acpr_mosaic_trust_v3_icdor_direct_image":
        raise ICDORAuditError("IC-DOR source manifest target branch is invalid")
    return {"pass": True, "path": str(path), **payload}


def _require_source_tokens(path: Path, required: Iterable[str], forbidden: Iterable[str] = ()) -> dict[str, Any]:
    if not path.is_file():
        raise ICDORAuditError(f"required audit source is missing: {path}")
    source = path.read_text(encoding="utf-8")
    missing = [token for token in required if token not in source]
    present = [token for token in forbidden if token in source]
    if missing or present:
        raise ICDORAuditError(f"source contract failed for {path.name}; missing={missing}, forbidden={present}")
    return {"path": str(path), "sha256": sha256_file(path), "required_tokens": list(required)}


def functional_hard_gates(
    root: Path,
    config: Mapping[str, Any],
    dynamic: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Derive every PASS from explicit source/config/dynamic evidence."""
    model_path = root / "fate_oia" / "models" / "acpr_mosaic_trust_icdor_model.py"
    trainer_path = root / "fate_oia" / "engine" / "train_acpr_mosaic_trust_icdor.py"
    evidence: dict[str, Any] = {}
    evidence["direct_image"] = {
        "config": config["experiment"].get("direct_image") is True,
        "feature_cache": config["backbone"].get("feature_cache"),
        "token_compression": config["backbone"].get("token_compression"),
        "dynamic_input_sensitivity": dynamic["input_sensitivity"],
        "source": _require_source_tokens(model_path, ("self.dino(images)",)),
    }
    if not evidence["direct_image"]["config"] or evidence["direct_image"]["feature_cache"] is not False or evidence["direct_image"]["token_compression"] != "none":
        raise ICDORAuditError("direct-image/no-cache/no-compression config gate failed")
    evidence["factor_certificate"] = {
        "builder": _require_source_tokens(
            root / "fate_oia" / "engine" / "build_mosaic_factor_certificate.py",
            ("build_factor_certificate", "source_split", "train_audit", "write_json"),
        ),
        "certificate_logic": _require_source_tokens(
            root / "fate_oia" / "models" / "mosaic_factor_certificate.py",
            ("bootstrap_lcb95", "reason_only", "certified", "abstained", "effective_count"),
        ),
    }
    evidence["edge_admission"] = _require_source_tokens(
        root / "fate_oia" / "engine" / "build_mosaic_edge_admission.py",
        ("source_split", "train_audit", "signed_effect_lcb95", "tes_lcb95", "isolated_edge_ap"),
    )
    evidence["action_firewall"] = {
        "cross_gradient": dynamic["gradient_firewall"]["reason_to_action_adapter"],
        "source": _require_source_tokens(model_path, ("factor_output[\"factor_features\"]", ".detach()")),
    }
    evidence["reason_firewall"] = {
        "cross_gradient": dynamic["gradient_firewall"]["action_to_reason_adapter"],
        "source": _require_source_tokens(model_path, ("self._detached_pyramid(reason_pyramid)",)),
    }
    if any(float(value) != 0.0 for value in dynamic["gradient_firewall"].values()):
        raise ICDORAuditError("dynamic action/reason gradient firewall failed")
    evidence["selective_observation"] = _require_source_tokens(
        root / "fate_oia" / "models" / "mosaic_icdor_observation_head.py",
        ("def posterior_from_observed_targets", "factor_visibility.detach()", "reason_logits_latent.detach()", "pi_min", "pi_max"),
    )
    evidence["calibration"] = {
        "config": {
            "train_calib_only": config["calibration"].get("train_calib_only"),
            "deploy_equation": config["calibration"].get("deploy_equation"),
            "test_oracle_diagnostic_only": config["calibration"].get("test_oracle_diagnostic_only"),
        },
        "source": _require_source_tokens(trainer_path, ("fit_icdor_calibration(", "source_split\": \"train_calib")),
    }
    if evidence["calibration"]["config"] != {
        "train_calib_only": True, "deploy_equation": "raw_minus_theta", "test_oracle_diagnostic_only": True
    }:
        raise ICDORAuditError("train-calib-only calibration gate failed")
    evidence["artifact_schema"] = _require_source_tokens(
        root / "fate_oia" / "utils" / "mosaic_icdor_artifacts.py",
        ("target_transfer_summary.json", "visual_audit_manifest.json", "gradient_ownership.jsonl", "strict_semantics"),
    )
    evidence["resume_integrity"] = _require_source_tokens(
        trainer_path,
        ("certificate_sha256", "edge_admission_sha256", "action_queue.state_dict()", "reason_queue.state_dict()", "pareto.state_dict()"),
    )
    evidence["visual_audit"] = _require_source_tokens(
        root / "fate_oia" / "engine" / "export_mosaic_trust_visual_audit.py",
        ("train_audit", "fixed_sample_ids", "matched_random_factor_mask_files", "action_support_mask"),
    )
    evidence["foreground_launcher"] = _require_source_tokens(
        root / "scripts" / "FATE_OIA_acpr_mosaic_trust_v3_icdor_foreground.ps1",
        ("Invoke-ForegroundPython", "--require_review_pass", "--runtime_selection", "--write_review_pass"),
        ("Start-Process", "Start-Job", "nohup", "scheduled task"),
    )
    return ({name: "PASS" for name in _REQUIRED_FUNCTIONAL_CHECKS}, evidence)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _git_tree(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _worktree_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return not result.stdout.strip()


def _tracked_source_manifest_sha256(root: Path) -> str:
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--full-tree", "HEAD"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    return hashlib.sha256(result.stdout.encode("utf-8")).hexdigest().upper()


def build_review_pass(
    audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
    pilot: Mapping[str, Any],
    *,
    runtime_sha256: str,
) -> dict[str, Any]:
    """Build a hash-bound REVIEW_PASS only from complete, non-pending evidence."""
    if audit.get("pass") is not True or audit.get("missing_items"):
        raise ICDORAuditError("implementation audit is not complete")
    if (
        audit.get("worktree_clean") is not True
        or not audit.get("git_tree")
        or not audit.get("source_manifest_sha256")
        or not audit.get("contract_manifest_sha256")
    ):
        raise ICDORAuditError("audited source tree is dirty or not hash-bound")
    checks = audit.get("functional_checks")
    if not isinstance(checks, Mapping):
        raise ICDORAuditError("functional checks are missing")
    failed = [name for name in _REQUIRED_FUNCTIONAL_CHECKS if checks.get(name) != "PASS"]
    if failed:
        raise ICDORAuditError(f"functional checks are not passing: {failed}")
    selected = runtime.get("selected")
    if runtime.get("pass") is not True or not isinstance(selected, Mapping) or selected.get("status") != "PASS":
        raise ICDORAuditError("runtime profile is not a real passing selection")
    pending = pilot.get("pending_artifacts")
    if pilot.get("pass") is not True or pilot.get("artifacts_complete") is not True:
        raise ICDORAuditError("pilot gate is not complete")
    if pending:
        raise ICDORAuditError(f"pilot contains pending artifacts: {pending}")
    gates = {name: "PASS" for name in _REQUIRED_FUNCTIONAL_CHECKS}
    gates.update({"runtime_profile": "PASS", "pilot_artifacts": "PASS"})
    return {
        "status": "PASS",
        "target_head": str(audit["git_head"]),
        "target_tree": str(audit["git_tree"]),
        "source_manifest_sha256": str(audit["source_manifest_sha256"]),
        "contract_manifest_sha256": str(audit["contract_manifest_sha256"]),
        "resolved_config_sha256": str(audit["config_sha256"]),
        "runtime_selection_sha256": runtime_sha256,
        "gates": gates,
        "audit_sha256": hashlib.sha256(
            json.dumps(dict(audit), sort_keys=True).encode("utf-8")
        ).hexdigest().upper(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed IC-DOR source and protocol audit")
    parser.add_argument("--config", required=True)
    parser.add_argument("--worktree_root", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--review_pass")
    parser.add_argument("--runtime_selection")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fail_closed", action="store_true")
    parser.add_argument("--write_review_pass", action="store_true")
    parser.add_argument("--pilot_gate")
    parser.add_argument("--max_audit_samples", type=int, default=2)
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise ICDORAuditError(f"config is missing: {config_path}")
    root = Path(args.worktree_root).resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ICDORAuditError("real-DINO dynamic audit requested CUDA but CUDA is unavailable")
    # Lazy imports avoid a circular import during trainer unit tests.
    from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
        build_icdor_loaders,
        build_icdor_model,
        build_icdor_parameter_ownership,
        load_config,
    )
    config = load_config(config_path)
    _, audit_loader, _, _, split_stats = build_icdor_loaders(
        config, Path(args.output_dir) / "dynamic_data", batch_size=1, num_workers=0,
        max_train_samples=1, max_audit_samples=args.max_audit_samples,
        max_calib_samples=1, max_test_samples=1,
    )
    first_batch = next(iter(audit_loader))
    images = first_batch.get("image")
    if not isinstance(images, torch.Tensor):
        raise ICDORAuditError("real train_audit loader did not return image tensors")
    model = build_icdor_model(config).to(device)
    ownership, _ = build_icdor_parameter_ownership(model)
    dynamic = verify_dynamic_forward_and_gradients(model, images.to(device))
    functional, functional_evidence = functional_hard_gates(root, config, dynamic)
    clean = _worktree_clean(root)
    contract_manifest = root / ".review" / "icdor_source_manifest.json"
    result = {
        "pass": clean,
        "git_head": _git_head(root),
        "git_tree": _git_tree(root),
        "worktree_clean": clean,
        "source_manifest_sha256": _tracked_source_manifest_sha256(root),
        "contract_manifest_sha256": sha256_file(contract_manifest),
        "config_sha256": sha256_file(config_path),
        "source": source_hard_gate(root),
        "source_manifest": verify_source_manifest(root),
        "dynamic_forward": dynamic,
        "functional_checks": functional,
        "functional_evidence": functional_evidence,
        "parameter_ownership_count": len(ownership),
        "split_protocol": split_stats,
        "checked_files": [str(config_path), str(root / "fate_oia" / "models" / "acpr_mosaic_trust_icdor_model.py")],
        "missing_items": [],
        "warnings": [] if clean else ["worktree is dirty; REVIEW_PASS is forbidden until code is committed"],
    }
    if args.review_pass or args.runtime_selection:
        if not args.review_pass or not args.runtime_selection:
            raise ICDORAuditError("review_pass and runtime_selection must be supplied together")
        result["review"] = protocol_hard_gate(
            args.review_pass,
            target_head=result["git_head"],
            config_sha256=result["config_sha256"],
            runtime_sha256=sha256_file(args.runtime_selection),
            required_gates=tuple(_REQUIRED_FUNCTIONAL_CHECKS) + ("runtime_profile", "pilot_artifacts"),
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "icdor_source_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.fail_closed and result.get("pass") is not True:
        raise ICDORAuditError("fail-closed audit rejected a dirty or incomplete source tree")
    if args.write_review_pass:
        if not args.runtime_selection or not args.pilot_gate:
            raise ICDORAuditError("write_review_pass requires runtime_selection and pilot_gate")
        runtime = json.loads(Path(args.runtime_selection).read_text(encoding="utf-8"))
        pilot = json.loads(Path(args.pilot_gate).read_text(encoding="utf-8"))
        review = build_review_pass(result, runtime, pilot, runtime_sha256=sha256_file(args.runtime_selection))
        review_path = output / "acpr_mosaic_trust_v3_icdor_REVIEW_PASS.json"
        review_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

