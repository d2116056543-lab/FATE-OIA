import torch

from fate_oia.engine.train_acpr_meter_oia import _choose_counterfactual_factors
from fate_oia.losses.meter_counterfactual_losses import meter_counterfactual_loss
from fate_oia.models.meter_signed_factors import METERsignedFactors


def test_counterfactual_loss_has_selected_control_and_target_terms() -> None:
    values = [torch.tensor([0.2, 0.3]), torch.tensor([0.0, 0.1]), torch.tensor([-0.1, 0.0])]
    result = meter_counterfactual_loss(*values, values[0], values[0], target_action_effect=values[0], wrong_action_effect=values[1])
    assert {"selected_control", "specificity", "direction", "total"} <= result.keys()
    assert torch.isfinite(result["total"])


def test_counterfactual_counter_selected_must_beat_matched_control() -> None:
    common = [
        torch.tensor([0.2]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.2]),
        torch.tensor([0.1]),
    ]
    good = meter_counterfactual_loss(
        *common,
        counter_control_effect=torch.tensor([0.0]),
        counter_action_effect=torch.tensor([0.2]),
        counter_control_action_effect=torch.tensor([0.0]),
    )
    bad = meter_counterfactual_loss(
        *common,
        counter_control_effect=torch.tensor([0.15]),
        counter_action_effect=torch.tensor([0.1]),
        counter_control_action_effect=torch.tensor([0.15]),
    )
    assert float(bad["selected_control"]) > float(good["selected_control"])
    assert float(bad["direction"]) > float(good["direction"])


def test_support_and_counter_factors_follow_signed_action_contribution() -> None:
    contributions = torch.tensor(
        [[[0.8, -0.7, 0.1], [0.0, 0.0, 0.0]]]
    )
    reliability = torch.ones(1, 3)
    support_score = torch.tensor([[0.9, 0.9, 0.9]])
    counter_score = torch.tensor([[0.9, 0.9, 0.9]])

    selected = _choose_counterfactual_factors(
        contributions,
        reliability,
        support_score,
        counter_score,
        torch.tensor([0]),
    )

    assert int(selected["support_factor"][0]) == 0
    assert int(selected["counter_factor"][0]) == 1
    assert bool(selected["support_valid"][0])
    assert bool(selected["counter_valid"][0])


def test_signed_factor_attribution_reconstructs_score_preactivation() -> None:
    torch.manual_seed(53)
    module = METERsignedFactors(dim=16, factor_dim=3, num_layers=3, rank=4)
    output = module(
        torch.randn(2, 3, 16),
        torch.randn(2, 3, 10, 16),
        progress=1.0,
    )

    support_preactivation = torch.log(torch.expm1(output["factor_support_score"]))
    counter_preactivation = torch.log(torch.expm1(output["factor_counter_score"]))
    torch.testing.assert_close(
        output["factor_support_attribution"].sum(-1)
        + output["factor_support_null_attribution"]
        + module.support_score_head.bias.view(1, 1),
        support_preactivation,
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        output["factor_counter_attribution"].sum(-1)
        + output["factor_counter_null_attribution"]
        + module.counter_score_head.bias.view(1, 1),
        counter_preactivation,
        atol=1e-5,
        rtol=1e-5,
    )
    assert (output["factor_support_attribution"] > 0).any()
    assert (output["factor_support_attribution"] < 0).any()
