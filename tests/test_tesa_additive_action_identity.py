import torch

from fate_oia.models.meter_semantic_action import FactorSpecificActionTransport


def test_action_is_exact_bounded_sum_and_nonowned_is_zero() -> None:
    torch.manual_seed(3)
    module = FactorSpecificActionTransport(dim=32)
    out = module(
        torch.randn(2, 4),
        torch.randn(2, 4, 32),
        torch.randn(2, 21, 32, requires_grad=True),
        torch.ones(2, 21),
        torch.tensor([1.0] * 14 + [0.0] + [1.0] * 5 + [0.0]),
        progress=1,
    )
    expected = out["action_logits_visual"] + out["action_factor_contributions"].sum(-1)
    torch.testing.assert_close(out["action_logits_final"], expected)
    deleted = out["action_logits_final"] - out["action_factor_contributions"][..., 3]
    torch.testing.assert_close(
        deleted,
        out["action_logits_visual"]
        + torch.cat(
            (
                out["action_factor_contributions"][..., :3],
                out["action_factor_contributions"][..., 4:],
            ),
            -1,
        ).sum(-1),
    )
    assert out["action_factor_contributions"][..., [14, 20]].eq(0).all()
    assert out["action_factor_weights"][..., [14, 20]].eq(0).all()
