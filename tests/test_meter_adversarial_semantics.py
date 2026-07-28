import inspect

import torch

from fate_oia.engine import train_acpr_meter_oia as trainer
from fate_oia.losses.meter_grounding_losses import meter_grounding_loss
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.optim.meter_meta_utility import METERMetaUtility


def _grounding_case(support_score: torch.Tensor, counter_score: torch.Tensor) -> tuple[dict, dict]:
    support_map = torch.tensor([[[0.7, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.7]]])
    counter_map = torch.tensor([[[0.1, 0.1, 0.1, 0.7], [0.7, 0.1, 0.1, 0.1]]])
    output = {
        "factor_support_map": support_map,
        "factor_counter_map": counter_map,
        "factor_support_score": support_score,
        "factor_counter_score": counter_score,
    }
    targets = {
        "factor_support_map": support_map.reshape(1, 2, 1, 4),
        "factor_counter_map": counter_map.reshape(1, 2, 1, 4),
        "factor_support_valid": torch.tensor([[True, False]]),
        "factor_counter_valid": torch.tensor([[False, True]]),
    }
    return output, targets


def test_grounding_evidence_score_rewards_signed_direction() -> None:
    aligned_output, targets = _grounding_case(
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
    )
    collapsed_output, _ = _grounding_case(
        torch.tensor([[0.5, 0.5]]),
        torch.tensor([[0.5, 0.5]]),
    )
    aligned = meter_grounding_loss(aligned_output, targets)
    collapsed = meter_grounding_loss(collapsed_output, targets)
    assert aligned["evidence"] < collapsed["evidence"]


def test_grounding_loss_contains_mirror_consistency() -> None:
    signature = inspect.signature(meter_grounding_loss)
    assert "mirror_pairs" in signature.parameters
    source = inspect.getsource(meter_grounding_loss)
    assert '"mirror"' in source


def test_grounding_and_counterfactual_use_independent_ramps() -> None:
    source = inspect.getsource(trainer._losses)
    assert "grounding_ramp_fraction" in source
    assert "counterfactual_ramp_fraction" in source


def test_counterfactual_uses_cumulative_mass_and_nonselected_neighbors() -> None:
    signature = inspect.signature(trainer._counterfactual_event)
    assert "selected_mass" in signature.parameters
    source = inspect.getsource(trainer._counterfactual_event)
    assert "_select_mass_mask" in source
    assert "_replace_selected_with_neighbor_mean" in source


def test_cumulative_mass_selection_and_neighbor_replacement_are_numerical() -> None:
    score = torch.tensor([[[0.40, 0.30, 0.20, 0.10]]])
    mask, count = trainer._select_mass_mask(
        score,
        torch.tensor([0]),
        selected_mass=0.60,
        minimum_patches=1,
        max_patches=4,
    )
    assert mask.tolist() == [[True, True, False, False]]
    assert count.tolist() == [2]
    tokens = torch.tensor([[[1.0], [2.0], [3.0], [9.0]]])
    selected = torch.tensor([[False, False, False, True]])
    replaced = trainer._replace_selected_with_neighbor_mean(tokens, selected, grid_hw=(2, 2))
    # The selected value 9.0 must not leak into its own replacement.
    assert torch.allclose(replaced[0, 3], torch.tensor([2.0]))


def test_pu_branch_cannot_update_meta_adapter() -> None:
    torch.manual_seed(17)
    model = METEROIAModel(use_mock_dino=True)
    output = model(torch.randn(1, 3, 360, 640), progress=1.0)
    assert "reason_logits_pu_private" in output
    output["reason_logits_pu_private"].square().mean().backward()
    meta_grads = [
        parameter.grad
        for parameter in model.signed_factors.meta_adapters.parameters()
        if parameter.requires_grad
    ]
    private_grads = [
        parameter.grad
        for parameter in model.reason_decoder.parameters()
        if parameter.requires_grad
    ]
    assert all(grad is None or float(grad.abs().sum()) == 0.0 for grad in meta_grads)
    assert any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in private_grads)


def test_pu_lambda_is_not_applied_twice_by_total_loss() -> None:
    source = inspect.getsource(trainer._losses)
    assert 'cfg["pu"].get("max_lambda"' not in source


def test_meta_utility_is_computed_per_selected_factor() -> None:
    utility = METERMetaUtility(
        factors=2,
        virtual_lr=0.1,
        ema_old_weight=0.0,
        ema_new_weight=1.0,
        lower=-0.01,
        upper=0.01,
    )
    parameters = {"weight": torch.zeros(2, 1, requires_grad=True)}

    def action_loss(candidate: dict[str, torch.Tensor]) -> torch.Tensor:
        weight = candidate["weight"].flatten()
        return (weight[0] - 1.0).square() + (weight[1] - 1.0).square()

    def reason_loss(
        candidate: dict[str, torch.Tensor],
        factor_id: int,
    ) -> torch.Tensor:
        weight = candidate["weight"].flatten()
        targets = weight.new_tensor([1.0, -1.0])
        return (weight[factor_id] - targets[factor_id]).square()

    event = utility.event(
        parameters,
        factor_ids=(0, 1),
        action_loss_fn=action_loss,
        reason_loss_fn=reason_loss,
        audit_action_loss_fn=action_loss,
    )
    assert event.relative_utility.shape == (2,)
    assert float(event.relative_utility[0]) > 0.0
    assert float(event.relative_utility[1]) < 0.0
