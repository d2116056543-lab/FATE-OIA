import torch

from fate_oia.losses.meter_reason_losses import (
    build_reason_supervision,
    reason_correction_sign_loss,
)
from fate_oia.models.meter_reason_decoder import METERPrivateReasonDecoder


def test_reason_correction_starts_as_exact_global_anchor_and_receives_gradient() -> None:
    torch.manual_seed(13)
    decoder = METERPrivateReasonDecoder(dim=4, reason_dim=3, action_dim=4)
    assert decoder.correction_vector.eq(0).all()
    output = decoder(
        reason_logits_calalign=torch.randn(2, 3),
        reason_nodes=torch.randn(2, 3, 4),
        factor_measurement_token=torch.randn(2, 3, 4),
        factor_reliability=torch.ones(2, 3),
        factor_groundable_mask=torch.ones(3),
        progress=1.0,
    )
    assert torch.allclose(output["reason_logits_final"], output["reason_logits_global"])
    output["reason_logits_final"].sum().backward()
    assert decoder.correction_vector.grad is not None
    assert decoder.correction_vector.grad.abs().sum() > 0
    assert decoder.global_delta_head.weight.grad is not None
    assert decoder.global_delta_head.weight.grad.abs().sum() > 0


def test_reason_correction_sign_balances_observed_positive_against_many_unknowns() -> None:
    target = torch.zeros(100, 1)
    target[0, 0] = 1.0
    supervision = build_reason_supervision(target, torch.zeros_like(target))
    correction = torch.full_like(target, -0.10)
    loss = reason_correction_sign_loss(correction, supervision, margin=0.05)
    assert torch.isclose(loss, torch.tensor(0.075), atol=1e-6)


def test_reason_correction_sign_handles_single_supervision_group_without_dilution() -> None:
    target = torch.zeros(5, 1)
    supervision = build_reason_supervision(target, torch.zeros_like(target))
    correction = torch.full_like(target, 0.10)
    assert torch.isclose(
        reason_correction_sign_loss(correction, supervision, margin=0.05),
        torch.tensor(0.15),
        atol=1e-6,
    )
