from __future__ import annotations

import argparse
import ast
import compileall
import hashlib
import inspect
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder
from fate_oia.engine.mosaic_schedule import mosaic_phase_controls
from fate_oia.engine.train_acpr_mosaic_ad import load_config
from fate_oia.losses.mosaic_posterior_ranking import posterior_weighted_reason_ranking_loss
from fate_oia.losses.mosaic_reason_observation_losses import build_mosaic_reason_loss
from fate_oia.models.acpr_mosaic_ad_model import MOSAICADModel
from fate_oia.models.mosaic_group_threshold import MOSAICGroupThresholdHead
from fate_oia.models.mosaic_native_semantics import load_mosaic_schema_bundle
from fate_oia.models.mosaic_selective_observation import MOSAICSelectiveObservationModel
from fate_oia.models.mosaic_state_composer import MOSAICSupportVetoComposer
from fate_oia.optim.mosaic_action_anchor import MOSAICActionAnchoredGradient
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue
from fate_oia.utils.mosaic_artifacts import validate_artifact_schema, write_json


# User-approved pre-full diagnostic scope. Formal training keeps its fixed seed.
PILOT_SEEDS = (20260710,)


GATE_FILES = {
    "code": "compile_test_gate.json",
    "schema": "schema_ontology_gate.json",
    "direct_image": "direct_image_gate.json",
    "typed_attention": "typed_attention_gate.json",
    "prototype": "prototype_gate.json",
    "visibility_presence": "visibility_presence_gate.json",
    "grounding_no_leakage": "grounding_no_leakage_gate.json",
    "state_monotonicity": "state_monotonicity_gate.json",
    "action_firewall": "action_firewall_gate.json",
    "label_decoder": "label_decoder_gate.json",
    "selective_observation": "selective_observation_math_gate.json",
    "synthetic_missing": "synthetic_missing_recovery_gate.json",
    "posterior_ranking": "posterior_ranking_gate.json",
    "action_anchor": "action_anchor_gate.json",
    "calibration": "calibration_gate.json",
    "schedule": "schedule_gate.json",
    "artifacts": "artifact_schema_gate.json",
    "runtime": "runtime_memory_gate.json",
    "pilot": "pilot_gate.json",
}

EXPECTED_FILES = (
    "configs/mosaic_label_schema.yaml",
    "configs/mosaic_observable_factors.yaml",
    "configs/mosaic_decision_states.yaml",
    "configs/mosaic_reason_observation.yaml",
    "configs/fate_oia_train_360x640_acpr_mosaic_ad_v1.yaml",
    "fate_oia/models/mosaic_native_semantics.py",
    "fate_oia/models/mosaic_visual_pyramid.py",
    "fate_oia/models/mosaic_geometry_typed_attention.py",
    "fate_oia/models/mosaic_observable_predicates.py",
    "fate_oia/models/mosaic_state_composer.py",
    "fate_oia/models/mosaic_sparse_label_decoder.py",
    "fate_oia/models/mosaic_action_decoder.py",
    "fate_oia/models/mosaic_reason_decoder.py",
    "fate_oia/models/mosaic_selective_observation.py",
    "fate_oia/models/mosaic_group_threshold.py",
    "fate_oia/models/acpr_mosaic_ad_model.py",
    "fate_oia/losses/mosaic_action_losses.py",
    "fate_oia/losses/mosaic_factor_losses.py",
    "fate_oia/losses/mosaic_reason_observation_losses.py",
    "fate_oia/losses/mosaic_posterior_ranking.py",
    "fate_oia/losses/mosaic_state_losses.py",
    "fate_oia/optim/mosaic_action_anchor.py",
    "fate_oia/optim/mosaic_soft_rank_queue.py",
    "fate_oia/datasets/mosaic_grounding_observations.py",
    "fate_oia/datasets/mosaic_multiview.py",
    "fate_oia/datasets/mosaic_train_calib_split.py",
    "fate_oia/engine/train_acpr_mosaic_ad.py",
    "fate_oia/engine/eval_acpr_mosaic_ad.py",
    "fate_oia/engine/profile_acpr_mosaic_ad.py",
    "fate_oia/engine/audit_acpr_mosaic_ad.py",
    "fate_oia/engine/export_mosaic_visual_audit.py",
    "fate_oia/engine/build_mosaic_ablation_table.py",
    "scripts/FATE_OIA_acpr_mosaic_ad_v1_foreground.ps1",
    ".codex/skills/acpr-mosaic-ad-implementation-audit/SKILL.md",
)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _pytest(repo_root: Path) -> tuple[bool, int, str]:
    files = sorted((repo_root / "tests").glob("test_mosaic_*.py"))
    command = [__import__("sys").executable, "-m", "pytest", *map(str, files), "-q"]
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    output = completed.stdout + completed.stderr
    count = 0
    for token in output.replace("\n", " ").split():
        if token.isdigit():
            count = max(count, int(token))
    return completed.returncode == 0, count, output[-4000:]


def _forbidden_ast(repo_root: Path) -> dict[str, Any]:
    formal_files = sorted(
        {
            *repo_root.glob("fate_oia/models/mosaic_*.py"),
            repo_root / "fate_oia/models/acpr_mosaic_ad_model.py",
            *repo_root.glob("fate_oia/losses/mosaic_*.py"),
            *repo_root.glob("fate_oia/optim/mosaic_*.py"),
            *repo_root.glob("fate_oia/datasets/mosaic_*.py"),
            repo_root / "fate_oia/engine/train_acpr_mosaic_ad.py",
            repo_root / "fate_oia/engine/eval_acpr_mosaic_ad.py",
            repo_root / "fate_oia/engine/profile_acpr_mosaic_ad.py",
            repo_root / "fate_oia/engine/export_mosaic_visual_audit.py",
        }
    )
    forbidden_calls = {
        "ACPROIAModel", "ACPRLabelTrunk", "ACPRPredicateReasoner", "ACPRPairMemory",
        "StartProcess", "StartJob",
    }
    forbidden_text = (
        "RunC",
        "cached_logits",
        "ACPROIAModel(",
        "ACPRLabelTrunk(",
        "ReasonToAction",
        "reason_to_action",
        "action_set_affects_final_action: true",
        "cooccurrence graph",
        "label graph",
    )
    findings = []
    for path in formal_files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else (
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                if name in forbidden_calls:
                    findings.append({"file": str(path), "line": node.lineno, "call": name})
        for line_number, line in enumerate(source.splitlines(), start=1):
            for pattern in forbidden_text:
                if pattern in line:
                    findings.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "match": pattern,
                            "disposition": "active_formal_path_forbidden",
                        }
                    )
    return {"pass": not findings, "findings": findings, "checked_files": [str(path) for path in formal_files]}


def _required_files_gate(repo_root: Path) -> dict[str, Any]:
    missing = [name for name in EXPECTED_FILES if not (repo_root / name).is_file()]
    test_files = sorted(path.name for path in (repo_root / "tests").glob("test_mosaic_*.py"))
    required_tests = {
        "test_mosaic_label_schema.py",
        "test_mosaic_visual_pyramid.py",
        "test_mosaic_geometry_typed_attention.py",
        "test_mosaic_prototype_routing.py",
        "test_mosaic_observable_predicates.py",
        "test_mosaic_grounding_observations.py",
        "test_mosaic_state_composer.py",
        "test_mosaic_action_firewall.py",
        "test_mosaic_reason_decoder.py",
        "test_mosaic_selective_observation.py",
        "test_mosaic_posterior_ranking.py",
        "test_mosaic_action_anchor.py",
        "test_mosaic_group_threshold.py",
        "test_mosaic_no_test_leakage.py",
        "test_mosaic_training_schedule.py",
        "test_mosaic_artifact_schema.py",
        "test_mosaic_memory_contract.py",
    }
    missing_tests = sorted(required_tests - set(test_files))
    return {
        "pass": not missing and not missing_tests,
        "checked_files": list(EXPECTED_FILES),
        "missing_files": missing,
        "missing_tests": missing_tests,
        "discovered_mosaic_tests": test_files,
    }


def _worktree_context_gate(
    repo_root: Path,
    *,
    branch: str,
    require_manifest_path_match: bool,
) -> dict[str, Any]:
    manifests = {
        "source_branch": repo_root / ".review/source_branch.txt",
        "source_commit": repo_root / ".review/source_commit.txt",
        "new_branch": repo_root / ".review/new_branch.txt",
        "worktree_path": repo_root / ".review/worktree_path.txt",
    }
    missing = [name for name, path in manifests.items() if not path.is_file()]
    values = {
        name: path.read_text(encoding="utf-8").strip()
        for name, path in manifests.items()
        if path.is_file()
    }
    source_is_ancestor = False
    if values.get("source_commit"):
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", values["source_commit"], "HEAD"],
            cwd=repo_root,
            capture_output=True,
        )
        source_is_ancestor = completed.returncode == 0
    actual_path = str(repo_root.resolve())
    original_path = str((repo_root.parent / "fate_oia_acpr_calalign_v1_2_worktree").resolve())
    path_match = actual_path.casefold() == values.get("worktree_path", "").casefold()
    pass_value = (
        not missing
        and branch == "acpr_mosaic_ad_v1_direct_image"
        and values.get("source_branch") == "github/acpr_calalign_v1_2"
        and values.get("new_branch") == "acpr_mosaic_ad_v1_direct_image"
        and actual_path.casefold() != original_path.casefold()
        and source_is_ancestor
        and (path_match or not require_manifest_path_match)
    )
    return {
        "pass": pass_value,
        "branch": branch,
        "actual_path": actual_path,
        "manifest_values": values,
        "missing_manifests": missing,
        "source_commit_is_ancestor": source_is_ancestor,
        "manifest_path_matches": path_match,
        "manifest_path_match_required": require_manifest_path_match,
    }
def _dynamic_model_gates(repo_root: Path, device: torch.device, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    model = MOSAICADModel(
        config_root=repo_root / "configs",
        backbone_arch=str(config["backbone"]["arch"]),
        backbone_patch_size=int(config["backbone"]["patch_size"]),
        selected_layers=tuple(int(value) for value in config["backbone"]["selected_layers"]),
        checkpoint_key=str(config["backbone"]["checkpoint_key"]),
        pretrained_weights=config["backbone"]["pretrained_weights"],
        use_mock_dino=device.type != "cuda",
        decoder_layers=1,
        self_attention_heads=4,
        highres_topk=16,
        midres_topk=8,
        anchors_per_factor=int(config["model"]["anchors_per_factor"]),
        typed_attention_heads=int(config["model"]["typed_attention_heads"]),
        point_samples=int(config["model"]["point_samples"]),
        curve_samples=int(config["model"]["curve_samples"]),
        region_samples=int(config["model"]["region_samples"]),
        spatial_prior_scale_init=float(config["model"]["spatial_prior_scale_init"]),
        spatial_prior_scale_max=float(config["model"]["spatial_prior_scale_max"]),
        spatial_prior_dropout=float(config["model"]["spatial_prior_dropout"]),
        content_temperature_init=float(config["model"]["content_temperature_init"]),
        state_residual_cap=float(config["model"]["state_residual_cap"]),
    ).to(device)
    model.train()
    model.set_phase_controls(
        state_residual_scale=0.10,
        action_state_gate_cap=0.15,
        reason_state_contribution_cap=0.10,
    )
    images = torch.randn(1, 3, 360, 640, device=device)
    output = model(images, return_masks=True)
    shape_pass = (
        output["action_logits_raw"].shape == (1, 4)
        and output["reason_logits_latent"].shape == (1, 21)
        and output["factor_soft_masks"].shape == (1, 24, 45, 80)
    )
    loss = output["factor_presence_logits"].float().square().mean()
    loss.backward()
    typed = model.observable_predicates.typed_attention
    typed_gradients = {
        name: float(parameter.grad.abs().sum().detach().cpu()) if parameter.grad is not None else 0.0
        for name, parameter in typed.named_parameters()
    }
    prototype_gradient = model.observable_predicates.prototype_bank.prototypes.grad
    prototype_grad_norm = float(prototype_gradient.abs().sum().detach().cpu()) if prototype_gradient is not None else 0.0

    model.eval()
    with torch.no_grad():
        before = model(images)["action_logits_raw"].clone()
        for parameter in model.reason_decoder.parameters():
            parameter.add_(torch.randn_like(parameter) * 3.0)
        after = model(images)["action_logits_raw"]
    action_firewall = torch.equal(before, after)
    grounding_builder = MOSAICGroundingObservationBuilder(model.schema_bundle["factors"])
    with torch.no_grad():
        leakage_before = model(images)
        grounding_builder(
            torch.zeros(1, 21, device=device),
            [{"image_size": (720, 1280), "objects": [], "lanes": [], "drivable_mask": None}],
            split="train",
        )
        leakage_after = model(images)
    grounding_invariant = torch.equal(
        leakage_before["action_logits_raw"], leakage_after["action_logits_raw"]
    ) and torch.equal(
        leakage_before["reason_logits_latent"], leakage_after["reason_logits_latent"]
    )
    signature = list(inspect.signature(MOSAICADModel.forward).parameters)
    return {
        "direct_image": {
            "pass": device.type == "cuda" and shape_pass and signature == ["self", "images", "prior_mode", "return_masks"],
            "forward_signature": signature,
            "real_dino": device.type == "cuda",
            "dino_frozen": all(not parameter.requires_grad for parameter in model.dino.parameters()),
        },
        "typed_attention": {
            "pass": all(value > 0 and torch.isfinite(torch.tensor(value)) for value in typed_gradients.values()),
            "gradient_norms": typed_gradients,
            "point_called": typed.point_indices.numel() > 0,
            "curve_called": typed.curve_indices.numel() > 0,
            "region_called": typed.region_indices.numel() > 0,
        },
        "prototype": {
            "pass": prototype_grad_norm > 0,
            "prototype_grad_norm": prototype_grad_norm,
            "prototype_count_min": int(model.observable_predicates.prototype_bank.prototype_valid_mask.sum(-1).min()),
        },
        "visibility_presence": {
            "pass": not torch.equal(output["factor_presence_logits"], output["factor_visibility_logits"]),
            "separate_heads": id(model.observable_predicates.presence_head) != id(model.observable_predicates.visibility_head),
        },
        "action_firewall": {
            "pass": action_firewall
            and not {"action", "reason", "labels", "geometry", "threshold", "metadata"} & set(signature),
            "reason_decoder_mutation_bitwise_equal": action_firewall,
            "forward_forbidden_inputs_absent": not {
                "action", "reason", "labels", "geometry", "threshold", "metadata"
            } & set(signature),
            "posterior_and_propensity_external_to_model": not hasattr(model, "selective_observation"),
        },
        "grounding_no_leakage": {
            "pass": grounding_invariant and "geometry" not in inspect.signature(MOSAICADModel.forward).parameters,
            "action_logits_bitwise_equal": torch.equal(
                leakage_before["action_logits_raw"], leakage_after["action_logits_raw"]
            ),
            "reason_logits_bitwise_equal": torch.equal(
                leakage_before["reason_logits_latent"], leakage_after["reason_logits_latent"]
            ),
            "geometry_forward_argument_absent": "geometry" not in inspect.signature(MOSAICADModel.forward).parameters,
        },
        "label_decoder": {
            "pass": output["action_nodes"].shape == (1, 4, model.dino.dim)
            and output["reason_nodes_visual"].shape == (1, 21, model.dino.dim),
            "action_nodes_shape": list(output["action_nodes"].shape),
            "reason_nodes_shape": list(output["reason_nodes_visual"].shape),
        },
    }


def _selective_gate(bundle: dict[str, Any]) -> dict[str, Any]:
    factor_names = [factor["name"] for factor in bundle["factors"]]
    module = MOSAICSelectiveObservationModel(factor_names, bundle["reason_observation"])
    logits = torch.tensor([[0.3] * 21], requires_grad=True)
    observed = torch.zeros_like(logits)
    visibility = torch.full((1, len(factor_names)), 0.7)
    uncertainty = torch.full_like(visibility, 0.2)
    output = module(logits, observed, visibility, uncertainty)
    pi = output["reason_propensity"]
    epsilon = output["reason_false_positive_rate"]
    p = torch.sigmoid(logits)
    expected_observation = pi * p + epsilon.unsqueeze(0) * (1.0 - p)
    expected_q = p * (1.0 - pi) / (
        p * (1.0 - pi) + (1.0 - p) * (1.0 - epsilon).unsqueeze(0)
    )
    math_pass = torch.allclose(output["reason_observation_prob"], expected_observation, atol=1e-6)
    math_pass = math_pass and torch.allclose(output["reason_latent_posterior"], expected_q, atol=1e-6)
    losses = build_mosaic_reason_loss(
        logits,
        observed,
        output["reason_observation_prob"],
        output["reason_latent_posterior"],
        output["reason_latent_posterior_live"],
        output["reason_propensity"],
        torch.zeros_like(observed, dtype=torch.bool),
        propensity_visibility_slopes=output["propensity_visibility_slopes"],
        propensity_uncertainty_slopes=output["propensity_uncertainty_slopes"],
        propensity_pi_min=output["propensity_pi_min"],
        propensity_pi_max=output["propensity_pi_max"],
        reason_false_positive_rate=output["reason_false_positive_rate"],
        reason_false_positive_max=output["reason_false_positive_max"],
    )
    expected_slope = (
        output["propensity_visibility_slopes"].square()
        + output["propensity_uncertainty_slopes"].square()
    ).mean()
    expected_boundary = (
        torch.relu(0.05 - (pi - output["propensity_pi_min"])).square()
        + torch.relu(0.05 - (output["propensity_pi_max"] - pi)).square()
    ).mean()
    epsilon_ratio = torch.where(
        output["reason_false_positive_max"] > 0,
        epsilon / output["reason_false_positive_max"].clamp_min(1e-12),
        torch.zeros_like(epsilon),
    )
    expected_propensity_regularizer = expected_slope + 0.10 * expected_boundary + 0.10 * epsilon_ratio.square().mean()
    regularizer_pass = torch.allclose(
        losses["loss_propensity_regularization"], expected_propensity_regularizer, atol=1e-7
    )
    expected_latent_range = torch.relu(torch.sigmoid(logits).mean(0) - 0.02).square().mean()
    prevalence_pass = torch.allclose(losses["loss_latent_rate_range"], expected_latent_range, atol=1e-7)
    output["reason_observation_prob"].sum().backward()
    return {
        "pass": bool(math_pass)
        and bool(regularizer_pass)
        and bool(prevalence_pass)
        and logits.grad is not None
        and torch.isfinite(logits.grad).all().item(),
        "pi_min": float(pi.min().detach()), "pi_max": float(pi.max().detach()),
        "epsilon_max": float(epsilon.max().detach()), "posterior_detached": not output["reason_latent_posterior"].requires_grad,
        "propensity_regularizer_exact": bool(regularizer_pass),
        "squared_prevalence_hinge_exact": bool(prevalence_pass),
    }


def _action_anchor_gate() -> dict[str, Any]:
    shared = torch.nn.Parameter(torch.zeros(2))
    action = torch.nn.Parameter(torch.zeros(1))
    explanation = torch.nn.Parameter(torch.zeros(1))
    helper = MOSAICActionAnchoredGradient(aux_shared_lambda_max=0.25, action_anchor_kappa=0.70)
    for action_vector, explanation_vector in zip(
        (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        (torch.tensor([-10.0, 0.0]), torch.tensor([0.0, -10.0])),
    ):
        helper.accumulate(
            (shared * action_vector).sum() + action.sum() * 0,
            (shared * explanation_vector).sum() + explanation.sum() * 0,
            [shared], [action], [explanation], loss_scale=1.0,
        )
    stats = helper.finalize(step=0)
    aggregate_action = torch.tensor([1.0, 1.0])
    aggregate_lhs = torch.dot(shared.grad, aggregate_action)
    aggregate_rhs = 0.70 * aggregate_action.square().sum()
    return {
        "pass": stats["constraint_pass"] and shared.grad is not None and bool(aggregate_lhs + 1e-6 >= aggregate_rhs),
        "aggregate_lhs": float(aggregate_lhs), "aggregate_rhs": float(aggregate_rhs), **stats,
    }


def _state_monotonicity_gate(bundle: dict[str, Any]) -> dict[str, Any]:
    factor_names = [factor["name"] for factor in bundle["factors"]]
    composer = MOSAICSupportVetoComposer(factor_names, bundle["states"], dim=8)
    state_names = list(composer.state_names)
    checks = []
    for factor_name, state_name, direction in (
        ("left_solid_boundary_visible", "left_veto", "increase"),
        ("left_drivable_visible", "left_affordance", "increase"),
        ("front_vehicle_near", "stop_obligation", "increase"),
        ("center_corridor_occupied", "forward_feasible", "decrease"),
    ):
        positive = torch.full((2, len(factor_names)), 0.25)
        negative = torch.full_like(positive, 0.20)
        uncertainty = torch.full_like(positive, 0.30)
        context = torch.zeros(2, 8, 12, 20)
        factor_index = factor_names.index(factor_name)
        state_index = state_names.index(state_name)
        positive[:, factor_index] = 0.05
        baseline = composer(positive, negative, uncertainty, context, residual_scale=0.0)["decision_state_prob"]
        positive[:, factor_index] = 0.95
        changed = composer(positive, negative, uncertainty, context, residual_scale=0.0)["decision_state_prob"]
        if direction == "increase":
            passed = bool(torch.all(changed[:, state_index] >= baseline[:, state_index])) and bool(
                torch.any(changed[:, state_index] > baseline[:, state_index])
            )
        else:
            passed = bool(torch.all(changed[:, state_index] <= baseline[:, state_index])) and bool(
                torch.any(changed[:, state_index] < baseline[:, state_index])
            )
        checks.append(
            {
                "factor": factor_name,
                "state": state_name,
                "direction": direction,
                "pass": passed,
                "baseline_mean": float(baseline[:, state_index].mean().detach()),
                "changed_mean": float(changed[:, state_index].mean().detach()),
            }
        )
    return {"pass": all(check["pass"] for check in checks), "interventions": checks}


def _posterior_ranking_gate() -> dict[str, Any]:
    queue = MOSAICSoftRankQueue(2, capacity=4)
    history_logits = torch.tensor([[0.0, -0.5]])
    history_q = torch.tensor([[0.2, 0.8]])
    queue.enqueue(history_logits, history_q, ["history"])
    current_logits = torch.tensor([[1.0, 0.5]], requires_grad=True)
    current_q = torch.tensor([[0.9, 0.4]], requires_grad=True)
    loss, stats = posterior_weighted_reason_ranking_loss(
        current_logits, current_q, ["current"], queue
    )
    weights = current_q.detach() * (1.0 - history_q)
    expected = (
        weights * torch.nn.functional.softplus(-(current_logits - history_logits))
    ).sum() / weights.sum()
    loss.backward()
    snapshot = queue.snapshot()
    return {
        "pass": torch.allclose(loss.detach(), expected.detach())
        and current_logits.grad is not None
        and current_q.grad is None
        and not snapshot["logits"].requires_grad
        and not snapshot["targets"].requires_grad,
        "loss": float(loss.detach()),
        "expected": float(expected.detach()),
        "pair_weight_sum": float(stats["pair_weight_sum"]),
        "soft_q_not_thresholded": True,
        "queue_detached": not snapshot["logits"].requires_grad and not snapshot["targets"].requires_grad,
    }


def _schedule_gate(repo_root: Path) -> dict[str, Any]:
    controls = [mosaic_phase_controls(epoch) for epoch in range(15)]
    expected = {
        0: (0.0, 0.0, 0.0, False, False, False),
        5: (0.10, 0.15, 0.10, False, False, False),
        6: (0.10, 0.15, 0.10, True, True, False),
        11: (0.10, 0.25, 0.20, True, True, False),
        12: (0.10, 0.25, 0.20, True, True, False),
        13: (0.10, 0.25, 0.20, False, False, True),
        14: (0.10, 0.25, 0.20, False, False, True),
    }
    measurements = {}
    pass_values = True
    for epoch, target in expected.items():
        control = controls[epoch]
        actual = (
            control.state_residual_scale,
            control.action_state_gate_cap,
            control.reason_state_contribution_cap,
            control.learned_propensity,
            control.posterior_enabled,
            control.calibration_only,
        )
        measurements[str(epoch)] = {"actual": actual, "expected": target, "pass": actual == target}
        pass_values = pass_values and actual == target
    trainer_source = (repo_root / "fate_oia/engine/train_acpr_mosaic_ad.py").read_text(encoding="utf-8")
    centralized = "mosaic_phase_controls(epoch)" in trainer_source and "epoch <= 2" not in trainer_source
    return {"pass": pass_values and centralized, "epochs": measurements, "centralized": centralized}


def _calibration_gate(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    head = MOSAICGroupThresholdHead(
        tail_reason_indices=calibration["tail_reason_indices"],
        label_delta_max=float(calibration["label_delta_max"]),
    )
    action = torch.randn(3, 4, requires_grad=True)
    reason = torch.randn(3, 21, requires_grad=True)
    output = head(action, reason)
    exact = torch.equal(output["action_logits_deploy"], action.detach() - output["threshold_logit"][:4])
    exact = exact and torch.equal(output["reason_logits_deploy"], reason.detach() - output["threshold_logit"][4:])
    output["logits_deploy"].sum().backward()
    from fate_oia.engine.train_acpr_mosaic_ad import fit_calibrator

    fit_source = inspect.getsource(fit_calibrator)
    train_calib_only = "test" not in fit_source.lower() and '"source": "train_calib"' in fit_source
    fixed_protocol = (
        calibration["steps_per_epoch"] == 100
        and calibration["batch_size"] == 256
        and calibration["surrogate_temperature"] == 0.20
        and calibration["soft_f1_weight"] == 1.00
        and calibration["bce_weight"] == 0.05
        and calibration["rate_weight"] == 0.02
        and calibration["delta_weight"] == 0.01
        and calibration["cardinality_weight"] == 0.02
        and calibration["label_delta_max"] == 1.0
        and "for step in range(max_steps)" in fit_source
        and "fixed_order" in fit_source
        and "calibration_objective" in fit_source
    )
    objective = head.calibration_objective(
        action.detach(),
        reason.detach(),
        torch.ones_like(action),
        torch.ones_like(reason),
        surrogate_temperature=float(calibration["surrogate_temperature"]),
        soft_f1_weight=float(calibration["soft_f1_weight"]),
        bce_weight=float(calibration["bce_weight"]),
        rate_weight=float(calibration["rate_weight"]),
        delta_weight=float(calibration["delta_weight"]),
        cardinality_weight=float(calibration["cardinality_weight"]),
    )
    objective_finite = bool(torch.isfinite(objective["loss_calibration_total"]))
    return {
        "pass": exact
        and action.grad is None
        and reason.grad is None
        and head.theta_group.grad is not None
        and train_calib_only
        and fixed_protocol
        and objective_finite,
        "deploy_exact": exact,
        "raw_logits_detached": action.grad is None and reason.grad is None,
        "train_calib_only": train_calib_only,
        "fixed_100_step_batch_256_protocol": fixed_protocol,
        "five_term_objective_finite": objective_finite,
    }


def _pilot_and_artifact_gates(
    pilot_dir: Path | None,
    artifact_smoke_dir: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if artifact_smoke_dir is not None and artifact_smoke_dir.exists():
        artifacts = validate_artifact_schema(
            artifact_smoke_dir,
            epochs=[0, 1],
            strict_semantics=True,
        )
        artifacts["source"] = str(artifact_smoke_dir)
    else:
        artifacts = {
            "pass": False,
            "status": "PENDING",
            "reason": "two-epoch artifact smoke not supplied",
        }
    if pilot_dir is None or not pilot_dir.exists():
        pending = {"pass": False, "status": "PENDING", "reason": "configured pilot seed not supplied"}
        return artifacts, pending, pending.copy()
    seed_dirs = [pilot_dir / f"seed_{seed}" for seed in PILOT_SEEDS]
    pilot_artifact_results = {
        str(path): validate_artifact_schema(
            path,
            epochs=list(range(8)),
            strict_semantics=True,
        )
        if path.exists()
        else {"pass": False, "missing": [str(path)]}
        for path in seed_dirs
    }
    visual_results = []
    recovery_results = []
    metric_results = []
    seed_checks = []
    for path in seed_dirs:
        visual_path = path / "visual_audit" / "summary.json"
        recovery_path = path / "epoch_007" / "posterior_recovery_stats.jsonl"
        metrics_path = path / "epoch_007" / "metrics_summary.json"
        if visual_path.exists():
            visual_results.append(json.loads(visual_path.read_text(encoding="utf-8")))
        if recovery_path.exists():
            rows = [json.loads(line) for line in recovery_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            recovery_results.append(next((row for row in reversed(rows) if row.get("summary")), {}))
        if metrics_path.exists():
            final_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metric_results.append(final_metrics)
            phase_b_path = path / "epoch_005" / "metrics_summary.json"
            action_branch_path = path / "epoch_007" / "action_branch_metrics.json"
            anchor_path = path / "epoch_007" / "action_anchor_stats.jsonl"
            if phase_b_path.exists() and action_branch_path.exists() and anchor_path.exists():
                phase_b = json.loads(phase_b_path.read_text(encoding="utf-8"))
                action_branch = json.loads(action_branch_path.read_text(encoding="utf-8"))
                anchor_rows = [json.loads(line) for line in anchor_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                active_anchor = [row for row in anchor_rows if row.get("available", True)]
                anchor_pass_rate = (
                    sum(bool(row.get("constraint_pass")) for row in active_anchor) / len(active_anchor)
                    if active_anchor else 0.0
                )
                seed_checks.append(
                    {
                        "action_no_selective_phase_collapse": final_metrics["raw"]["Act_mF1"] >= phase_b["raw"]["Act_mF1"] - 0.02,
                        "action_visual_competitive": action_branch["visual"]["Act_mF1"] >= final_metrics["raw"]["Act_mF1"] - 0.02,
                        "reason_map_no_collapse": final_metrics["raw"]["Exp_mAP"] >= phase_b["raw"]["Exp_mAP"] - 0.02,
                        "action_anchor_pass_rate": anchor_pass_rate,
                    }
                )
    pilot_pass = (
        len(metric_results) == len(PILOT_SEEDS)
        and len(seed_checks) == len(PILOT_SEEDS)
        and len(recovery_results) == len(PILOT_SEEDS)
        and all(result.get("pass") for result in pilot_artifact_results.values())
        and all(row.get("improvement", 0.0) > 0 for row in recovery_results)
        and all(
            row["action_no_selective_phase_collapse"]
            and row["action_visual_competitive"]
            and row["reason_map_no_collapse"]
            and row["action_anchor_pass_rate"] >= 0.95
            for row in seed_checks
        )
    )
    visual_pass = len(visual_results) == len(PILOT_SEEDS) and all(
        row.get("pass") is True
        and row.get("full_factor_metric", 0) > row.get("prior_only_factor_metric", float("inf"))
        and row.get("content_only_retention", 0) >= 0.70
        for row in visual_results
    )
    return (
        artifacts,
        {
            "pass": pilot_pass,
            "metrics": metric_results,
            "recovery": recovery_results,
            "seed_checks": seed_checks,
            "artifact_schema": pilot_artifact_results,
        },
        {"pass": visual_pass, "visual_results": visual_results},
    )


def _runtime_gate(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {"pass": False, "status": "PENDING", "reason": "runtime profiler not complete"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "batch_size",
        "grad_accum",
        "num_workers",
        "median_step_ms",
        "p95_step_ms",
        "samples_per_sec",
        "max_allocated_gb",
        "max_reserved_gb",
        "cuda_retries",
        "dataloader_stalls",
        "nan_count",
    }
    candidates = payload.get("candidates", [])
    passing = [record for record in candidates if record.get("status") == "PASS"]
    missing_fields = {
        str(index): sorted(required - set(record))
        for index, record in enumerate(candidates)
        if required - set(record)
    }
    selected = payload.get("selected", {})
    selected_match = next(
        (
            record
            for record in passing
            if all(record.get(key) == selected.get(key) for key in ("batch_size", "grad_accum", "num_workers"))
        ),
        None,
    )
    fastest = max(passing, key=lambda record: record["samples_per_sec"]) if passing else None
    fastest_selected = selected_match is not None and fastest is selected_match
    stability = payload.get("stability_probe", {})
    pass_value = (
        payload.get("pass") is True
        and payload.get("quick_diagnostic") is False
        and payload.get("phase_d_full_path") is True
        and not missing_fields
        and fastest_selected
        and selected_match["max_reserved_gb"] <= float(config["training"]["max_reserved_vram_gb"])
        and selected_match["cuda_retries"] == 0
        and selected_match["dataloader_stalls"] == 0
        and selected_match["nan_count"] == 0
        and stability.get("executed") is True
        and stability.get("pass") is True
        and float(stability.get("actual_seconds", 0.0)) >= 840.0
    )
    return {
        **payload,
        "pass": pass_value,
        "schema_missing_fields": missing_fields,
        "fastest_stable_candidate_selected": fastest_selected,
        "validated_by_audit": True,
    }


def _coverage_matrix(
    gate_status: dict[str, bool],
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    definitions = (
        ("isolated_worktree", "New worktree/branch and source manifests", ("code",)),
        ("independent_model", "Formal MOSAICADModel, not ACPROIAModel adapter", ("code", "direct_image")),
        ("no_old_checkpoint_or_cache", "No old checkpoint, cached logits, or compression", ("code",)),
        ("visibility_presence", "Distinct visibility and presence semantics", ("visibility_presence",)),
        ("typed_sparse_attention", "Point, curve, and region sparse samplers active", ("typed_attention",)),
        ("multi_prototype", "Independent prototype routing and content-over-prior audit", ("prototype",)),
        ("weak_spatial_prior", "Weak droppable priors with shuffle/content audit", ("prototype",)),
        ("support_veto", "Monotonic support-veto state composition", ("state_monotonicity",)),
        ("action_firewall", "Action independent of reason/posterior/propensity/geometry", ("action_firewall", "grounding_no_leakage")),
        ("latent_reason", "Latent reason visual/semantic decoder", ("label_decoder",)),
        ("exact_selective_observation", "Exact selective-observation posterior", ("selective_observation",)),
        ("synthetic_missing", "Synthetic missing-positive recovery", ("synthetic_missing",)),
        ("posterior_ranking", "Soft posterior-weighted cross-image ranking", ("posterior_ranking",)),
        ("action_anchor", "Action-anchored aggregate gradient update", ("action_anchor",)),
        ("calibration", "Independent train-calib group threshold", ("calibration",)),
        ("schedule", "Single canonical six-phase schedule", ("schedule",)),
        ("artifacts", "All root/epoch artifacts contain non-placeholder data", ("artifacts",)),
        ("runtime", "Fastest stable Phase-D runtime and 15-minute probe", ("runtime",)),
        ("pilot", "Configured-seed pilot and visual/content gates", ("pilot", "prototype", "synthetic_missing")),
    )
    runtime_gates = {"direct_image", "prototype", "synthetic_missing", "artifacts", "runtime", "pilot"}
    rows = []
    for item_id, requirement, gates in definitions:
        passed = all(gate_status.get(gate, False) for gate in gates)
        if passed:
            status = "VERIFIED"
        elif all(gate in runtime_gates or gate_status.get(gate, False) for gate in gates):
            status = "RUNTIME_PENDING"
        else:
            status = "BLOCKED"
        rows.append(
            {
                "id": item_id,
                "requirement": requirement,
                "must_have": True,
                "verification_gates": list(gates),
                "status": status,
            }
        )
    github_path = output_dir / "github_sync_pass.json"
    github_pass = False
    if github_path.exists():
        github_record = json.loads(github_path.read_text(encoding="utf-8"))
        github_pass = github_record.get("status") == "PASS" and github_record.get("local_head") == _git("rev-parse", "HEAD")
    rows.extend(
        (
            {
                "id": "github_sync",
                "requirement": "GitHub branch and fresh clone match local HEAD",
                "must_have": True,
                "verification_gates": ["github_sync_pass.json"],
                "status": "VERIFIED" if github_pass else "RUNTIME_PENDING",
            },
            {
                "id": "full_run",
                "requirement": "Foreground 15-epoch formal run",
                "must_have": True,
                "verification_gates": ["REVIEW_PASS", "GOAL_COMPLETED"],
                "status": "RUNTIME_PENDING",
            },
        )
    )
    return {
        "source_plan": "Codex_ACPR_MOSAIC_AD_V1_ImplementationPlan_20260710.md",
        "all_verified": all(row["status"] == "VERIFIED" for row in rows),
        "verified_count": sum(row["status"] == "VERIFIED" for row in rows),
        "runtime_pending_count": sum(row["status"] == "RUNTIME_PENDING" for row in rows),
        "blocked_count": sum(row["status"] == "BLOCKED" for row in rows),
        "items": rows,
    }


def audit(
    config_path: str,
    output_dir: str,
    *,
    device_name: str,
    pilot_dir: str | None,
    artifact_smoke_dir: str | None,
    write_review_pass: bool,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    git_head = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    status_lines = _git("status", "--short").splitlines()
    worktree_context = _worktree_context_gate(
        repo_root,
        branch=branch,
        require_manifest_path_match=write_review_pass,
    )
    write_json(
        output / "audit_context.json",
        {
            "git_head": git_head,
            "branch": branch,
            "source_branch": "acpr_calalign_v1_2",
            "config": str(Path(config_path).resolve()),
            "device": device_name,
            "dirty_paths": status_lines,
            "pilot_dir": str(Path(pilot_dir).resolve()) if pilot_dir else None,
            "artifact_smoke_dir": str(Path(artifact_smoke_dir).resolve()) if artifact_smoke_dir else None,
            "worktree_context": worktree_context,
        },
    )
    compile_pass = compileall.compile_dir(repo_root / "fate_oia", quiet=1)
    pytest_pass, tests_passed, pytest_tail = _pytest(repo_root)
    forbidden = _forbidden_ast(repo_root)
    required_files = _required_files_gate(repo_root)
    forbidden["config_checks"] = {
        "feature_cache_disabled": config["backbone"]["feature_cache"] is False,
        "token_compression_none": config["backbone"]["token_compression"] == "none",
        "test_only": config["evaluation"]["eval_splits"] == ["test"],
    }
    forbidden["pass"] = forbidden["pass"] and all(forbidden["config_checks"].values())
    write_json(output / "forbidden_pattern_scan.json", forbidden)
    code = {
        "pass": compile_pass
        and pytest_pass
        and forbidden["pass"]
        and required_files["pass"]
        and worktree_context["pass"],
        "compile": compile_pass,
        "pytest": pytest_pass,
        "tests_passed": tests_passed,
        "pytest_tail": pytest_tail,
        "forbidden": forbidden,
        "required_files": required_files,
        "worktree_context": worktree_context,
    }
    code["clean_worktree"] = not status_lines
    if write_review_pass and status_lines:
        code["pass"] = False
    bundle = load_mosaic_schema_bundle(repo_root / "configs")
    schema_fingerprints = {
        name: _sha256(repo_root / "configs" / name)
        for name in (
            "mosaic_label_schema.yaml", "mosaic_observable_factors.yaml",
            "mosaic_decision_states.yaml", "mosaic_reason_observation.yaml",
        )
    }
    schema = {"pass": len(bundle["factors"]) == 24 and len(bundle["states"]) == 8 and len(bundle["reason_observation"]) == 21, "factor_count": len(bundle["factors"]), "state_count": len(bundle["states"]), "reason_count": len(bundle["reason_observation"]), "fingerprints": schema_fingerprints}
    dynamic = _dynamic_model_gates(repo_root, torch.device(device_name), config)
    write_json(
        output / "forward_contract_gate.json",
        {
            "pass": dynamic["direct_image"]["forward_signature"]
            == ["self", "images", "prior_mode", "return_masks"],
            "signature": dynamic["direct_image"]["forward_signature"],
            "forbidden_inputs_absent": True,
        },
    )
    selective = _selective_gate(bundle)
    action_anchor = _action_anchor_gate()
    calibration = _calibration_gate(config)
    state_monotonicity = _state_monotonicity_gate(bundle)
    posterior_ranking = _posterior_ranking_gate()
    schedule = _schedule_gate(repo_root)
    artifacts, pilot, visual = _pilot_and_artifact_gates(
        Path(pilot_dir) if pilot_dir else None,
        Path(artifact_smoke_dir) if artifact_smoke_dir else None,
    )
    runtime_path = output / "mosaic_runtime_selection.json"
    runtime = _runtime_gate(runtime_path, config)
    gates = {
        "code": code,
        "schema": schema,
        **dynamic,
        "state_monotonicity": state_monotonicity,
        "selective_observation": selective,
        "synthetic_missing": {"pass": pilot["pass"], "pilot_recovery": pilot.get("recovery", [])},
        "posterior_ranking": posterior_ranking,
        "action_anchor": action_anchor,
        "calibration": calibration,
        "schedule": schedule,
        "artifacts": artifacts,
        "runtime": runtime,
        "pilot": pilot,
    }
    gates["prototype"]["content_over_prior"] = visual
    gates["prototype"]["pass"] = gates["prototype"]["pass"] and visual["pass"]
    for name, record in gates.items():
        write_json(output / GATE_FILES[name], record)
    gate_status = {name: bool(record.get("pass")) for name, record in gates.items()}
    write_json(output / "feature_coverage_matrix.json", _coverage_matrix(gate_status, repo_root=repo_root, output_dir=output))
    summary = {"pass": all(gate_status.values()), "git_head": git_head, "tests_passed": tests_passed, "gates": gate_status}
    write_json(output / "implementation_audit_ACPR_MOSAIC_AD_V1.json", summary)
    review_path = output / "acpr_mosaic_ad_v1_REVIEW_PASS.json"
    if review_path.exists():
        review_path.unlink()
    if write_review_pass:
        if not summary["pass"]:
            raise RuntimeError(f"MOSAIC audit cannot write REVIEW_PASS; failing gates: {[name for name, passed in gate_status.items() if not passed]}")
        review = {
            "status": "PASS", "git_head": summary["git_head"],
            "branch": "acpr_mosaic_ad_v1_direct_image", "source_branch": "acpr_calalign_v1_2",
            "config_hash": _sha256(config_path), "schema_hash": schema_fingerprints["mosaic_label_schema.yaml"],
            "runtime_selection_hash": _sha256(runtime_path), "tests_passed": tests_passed,
            "gates": gate_status, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(review_path, review)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=".review")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot_dir")
    parser.add_argument("--artifact_smoke_dir")
    parser.add_argument("--write_review_pass", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            audit(
                args.config,
                args.output_dir,
                device_name=args.device,
                pilot_dir=args.pilot_dir,
                artifact_smoke_dir=args.artifact_smoke_dir,
                write_review_pass=args.write_review_pass,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
