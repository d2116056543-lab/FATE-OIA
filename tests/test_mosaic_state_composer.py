from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from fate_oia.models.mosaic_native_semantics import load_mosaic_schema_bundle
from fate_oia.models.mosaic_state_composer import MOSAICSupportVetoComposer


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _composer(dim: int = 8) -> tuple[MOSAICSupportVetoComposer, list[str], list[str]]:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    factor_names = [factor["name"] for factor in bundle["factors"]]
    composer = MOSAICSupportVetoComposer(factor_names, bundle["states"], dim=dim)
    return composer, factor_names, list(composer.state_names)


def _inputs(factor_count: int, dim: int = 8) -> tuple[torch.Tensor, ...]:
    positive = torch.full((2, factor_count), 0.25)
    negative = torch.full((2, factor_count), 0.20)
    uncertainty = torch.full((2, factor_count), 0.30)
    context = torch.randn(2, dim, 12, 20)
    return positive, negative, uncertainty, context


def test_state_composer_returns_separate_support_veto_residual_and_uncertainty() -> None:
    composer, factors, states = _composer()
    output = composer(*_inputs(len(factors)), residual_scale=1.0)

    assert set(output) == {
        "decision_state_logits",
        "decision_state_prob",
        "decision_state_support",
        "decision_state_veto",
        "decision_state_residual",
        "decision_state_uncertainty",
        "state_stats",
    }
    for key in (
        "decision_state_logits",
        "decision_state_prob",
        "decision_state_support",
        "decision_state_veto",
        "decision_state_residual",
        "decision_state_uncertainty",
    ):
        assert output[key].shape == (2, len(states))
        assert torch.isfinite(output[key]).all()
    assert torch.all((output["decision_state_prob"] >= 0) & (output["decision_state_prob"] <= 1))
    assert torch.all((output["decision_state_support"] >= 0) & (output["decision_state_support"] <= 1))
    assert torch.all((output["decision_state_veto"] >= 0) & (output["decision_state_veto"] <= 1))
    assert output["decision_state_residual"].abs().max() <= 0.20


@pytest.mark.parametrize(
    ("factor_name", "state_name", "direction"),
    [
        ("left_solid_boundary_visible", "left_veto", "increase"),
        ("left_drivable_visible", "left_affordance", "increase"),
        ("front_vehicle_near", "stop_obligation", "increase"),
        ("center_corridor_occupied", "forward_feasible", "decrease"),
    ],
)
def test_support_veto_interventions_are_structurally_monotonic(
    factor_name: str,
    state_name: str,
    direction: str,
) -> None:
    composer, factors, states = _composer()
    positive, negative, uncertainty, context = _inputs(len(factors))
    positive[:, factors.index(factor_name)] = 0.05
    baseline = composer(positive, negative, uncertainty, context, residual_scale=0.0)["decision_state_prob"]
    positive[:, factors.index(factor_name)] = 0.95
    changed = composer(positive, negative, uncertainty, context, residual_scale=0.0)["decision_state_prob"]
    state_index = states.index(state_name)
    if direction == "increase":
        assert torch.all(changed[:, state_index] >= baseline[:, state_index])
        assert torch.any(changed[:, state_index] > baseline[:, state_index])
    else:
        assert torch.all(changed[:, state_index] <= baseline[:, state_index])
        assert torch.any(changed[:, state_index] < baseline[:, state_index])


def test_state_residual_scale_zero_is_exact_and_full_scale_is_bounded() -> None:
    composer, factors, _ = _composer()
    inputs = _inputs(len(factors))
    zero = composer(*inputs, residual_scale=0.0)
    full = composer(*inputs, residual_scale=1.0)
    assert torch.count_nonzero(zero["decision_state_residual"]) == 0
    assert full["decision_state_residual"].abs().max() <= composer.state_residual_cap
    with pytest.raises(ValueError, match="residual_scale"):
        composer(*inputs, residual_scale=1.1)


def test_zero_veto_is_a_neutral_penalty_not_a_positive_logit_boost() -> None:
    composer, factors, states = _composer()
    positive = torch.zeros(1, len(factors))
    positive[0, factors.index("left_solid_boundary_visible")] = 0.25
    zeros = torch.zeros_like(positive)
    output = composer(positive, zeros, zeros, torch.zeros(1, 8, 12, 20), residual_scale=0.0)
    state_index = states.index("left_veto")
    assert output["decision_state_veto"][0, state_index] == 0
    assert output["decision_state_prob"][0, state_index].item() == pytest.approx(
        output["decision_state_support"][0, state_index].item(), abs=1e-5
    )


def test_all_structural_and_visual_parameters_receive_gradients() -> None:
    torch.manual_seed(31)
    composer, factors, _ = _composer()
    inputs = list(_inputs(len(factors)))
    for value in inputs:
        value.requires_grad_()
    output = composer(*inputs, residual_scale=1.0)
    loss = output["decision_state_logits"].square().mean()
    loss.backward()

    assert all(parameter.grad is not None for parameter in composer.support_weights)
    assert all(parameter.grad is not None for parameter in composer.veto_weights)
    assert all(torch.isfinite(parameter.grad).all() for parameter in composer.support_weights)
    assert composer.raw_gamma.grad is not None and composer.raw_gamma.grad.abs().sum() > 0
    for parameter in (
        composer.state_queries,
        composer.context_key.weight,
        composer.context_value.weight,
        composer.residual_projection.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_negative_evidence_and_uncertainty_reduce_factor_support_and_receive_gradients() -> None:
    composer, factors, states = _composer()
    positive = torch.full((1, len(factors)), 0.25, requires_grad=True)
    negative = torch.zeros_like(positive, requires_grad=True)
    uncertainty = torch.zeros_like(positive, requires_grad=True)
    context = torch.zeros(1, 8, 12, 20)
    state_index = states.index("left_affordance")
    baseline = composer(positive, negative, uncertainty, context, residual_scale=0.0)["decision_state_prob"]
    changed_negative = negative.detach().clone()
    changed_negative[:, factors.index("left_drivable_visible")] = 0.9
    changed_uncertainty = uncertainty.detach().clone()
    changed_uncertainty[:, factors.index("left_turn_marking_visible")] = 0.9
    changed = composer(positive, changed_negative, changed_uncertainty, context, residual_scale=0.0)[
        "decision_state_prob"
    ]
    assert changed[0, state_index] < baseline[0, state_index]
    baseline[0, state_index].backward()
    assert negative.grad is not None and negative.grad.abs().sum() > 0
    assert uncertainty.grad is not None and uncertainty.grad.abs().sum() > 0


def test_large_veto_weight_remains_smooth_and_does_not_cut_gradients() -> None:
    composer, factors, states = _composer()
    with torch.no_grad():
        for weight in composer.veto_weights:
            weight.fill_(5.0)
    positive = torch.zeros(1, len(factors), requires_grad=True)
    positive.data[0, factors.index("left_drivable_visible")] = 0.8
    positive.data[0, factors.index("left_corridor_occupied")] = 0.9
    zeros = torch.zeros_like(positive)
    output = composer(positive, zeros, zeros, torch.zeros(1, 8, 12, 20), residual_scale=0.0)
    output["decision_state_prob"][0, states.index("left_affordance")].backward()
    veto_factor_gradient = positive.grad[0, factors.index("left_corridor_occupied")]
    assert torch.isfinite(veto_factor_gradient) and veto_factor_gradient.abs() > 0
    assert any(weight.grad is not None and weight.grad.abs().sum() > 0 for weight in composer.veto_weights)


def test_structural_weights_and_gamma_remain_bounded_under_extreme_raw_values() -> None:
    composer, _, _ = _composer()
    extreme = torch.tensor([-100.0, 0.0, 100.0])
    bounded = composer._bounded_nonnegative_weights(extreme)
    assert torch.all(bounded >= 0)
    assert torch.all(bounded < 1)
    with torch.no_grad():
        composer.raw_gamma.fill_(100.0)
    gamma = composer._bounded_gamma()
    assert torch.all(gamma >= 0)
    assert torch.all(gamma < 5.0)


def test_composer_rejects_unknown_references_cycles_and_shape_mismatch() -> None:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    factors = [factor["name"] for factor in bundle["factors"]]
    unknown = deepcopy(bundle["states"])
    unknown["front_risk"]["required_groups"][0]["any_of"] = ["unknown_factor"]
    with pytest.raises(ValueError, match="unknown factor/state"):
        MOSAICSupportVetoComposer(factors, unknown, dim=8)

    cycle = deepcopy(bundle["states"])
    cycle["front_risk"]["required_groups"][0]["any_of"] = ["stop_obligation"]
    with pytest.raises(ValueError, match="cycle"):
        MOSAICSupportVetoComposer(factors, cycle, dim=8)

    composer, factor_names, _ = _composer()
    positive, negative, uncertainty, context = _inputs(len(factor_names))
    with pytest.raises(ValueError, match="shape contract"):
        composer(positive[:, :-1], negative, uncertainty, context, residual_scale=0.0)


def test_state_composer_accepts_direct_bfloat16_context_and_evidence() -> None:
    composer, factors, _ = _composer()
    positive = torch.full((1, len(factors)), 0.3, dtype=torch.bfloat16, requires_grad=True)
    negative = torch.full_like(positive, 0.2)
    uncertainty = torch.full_like(positive, 0.1)
    context = torch.randn(1, 8, 12, 20, dtype=torch.bfloat16, requires_grad=True)
    output = composer(positive, negative, uncertainty, context, residual_scale=1.0)
    assert torch.isfinite(output["decision_state_logits"]).all()
    output["decision_state_logits"].float().sum().backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
