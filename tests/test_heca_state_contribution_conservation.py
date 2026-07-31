import torch

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_factor_contributions_reconstruct_bounded_action_delta() -> None:
    torch.manual_seed(9)
    module = StateConditionedActionCredit(dim=8, factor_dim=3, rank=4)
    out = module(
        action_logits_visual=torch.randn(2, 4),
        action_nodes=torch.randn(2, 4, 8),
        factor_action_bridge_token=torch.randn(2, 3, 8),
        factor_state_prob_credit=torch.softmax(torch.randn(2, 3, 3), -1),
        factor_reliability=torch.rand(2, 3),
        factor_action_ownership=torch.ones(3),
        progress=1.0,
    )
    summed = out["action_factor_contribution"].sum(-1)
    torch.testing.assert_close(summed, out["action_credit_sum"])
    expected = out["action_correction_kappa"] * torch.tanh(
        summed / out["action_correction_kappa"]
    )
    torch.testing.assert_close(out["action_evidence_delta_unramped"], expected)

