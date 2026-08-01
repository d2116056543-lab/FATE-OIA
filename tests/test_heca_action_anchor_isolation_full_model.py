import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_action_state_posterior_matches_reason_state_but_cannot_update_anchor_query() -> None:
    torch.manual_seed(31)
    model = METEROIAModel(dim=384, use_mock_dino=True)
    output = model(torch.randn(2, 3, 360, 640), progress=1.0)
    torch.testing.assert_close(
        output["factor_state_prob_action"], output["factor_state_prob"]
    )
    gradients = torch.autograd.grad(
        output["action_logits_final"].sum(),
        tuple(model.typed_factors.anchor_query.parameters()),
        allow_unused=True,
    )
    assert all(value is None or value.count_nonzero() == 0 for value in gradients)
