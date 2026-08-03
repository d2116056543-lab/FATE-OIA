import pytest
import torch

try:
    from fate_oia.losses.save_reason_losses import SAVE_REASON_LOSS_WEIGHTS, save_reason_loss
    from fate_oia.models.save_reason_decoder import SAVEPrivateReasonDecoder
except ImportError:
    SAVE_REASON_LOSS_WEIGHTS = None
    SAVEPrivateReasonDecoder = None
    save_reason_loss = None


def test_benchmark_reason_progress_zero_is_exactly_clean_anchor() -> None:
    if SAVEPrivateReasonDecoder is None:
        pytest.fail("SAVEPrivateReasonDecoder is not implemented")

    torch.manual_seed(51)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    clean = torch.randn(2, 3)
    output = decoder(
        reason_logits_clean=clean,
        global_field=torch.randn(2, 17, 8),
        detail_field=torch.randn(2, 17, 8),
        factor_measurement_token=torch.randn(2, 3, 8),
        factor_evidence_map=torch.rand(2, 3, 17),
        factor_reliability=torch.ones(2, 3),
        progress=0.0,
    )

    torch.testing.assert_close(
        output["reason_logits_benchmark"],
        clean,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        output["reason_logits_final"],
        clean,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(output["reason_logits_private_direct"] - clean) > 0


def test_progress_zero_weighted_private_direct_loss_still_trains_private_decoder() -> None:
    if SAVEPrivateReasonDecoder is None or save_reason_loss is None:
        pytest.fail("SAVE reason decoder or loss is not implemented")

    torch.manual_seed(52)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    clean = torch.randn(2, 3)
    output = decoder(
        reason_logits_clean=clean,
        global_field=torch.randn(2, 17, 8),
        detail_field=torch.randn(2, 17, 8),
        factor_measurement_token=torch.randn(2, 3, 8),
        factor_evidence_map=torch.rand(2, 3, 17),
        factor_reliability=torch.rand(2, 3),
        progress=0.0,
    )
    target = torch.randint(0, 2, (2, 3)).float()
    weights = {name: 0.0 for name in SAVE_REASON_LOSS_WEIGHTS}
    weights["private_direct"] = 0.35
    losses = save_reason_loss(output, target, weights=weights)

    torch.testing.assert_close(output["reason_logits_benchmark"], clean, atol=0.0, rtol=0.0)
    torch.testing.assert_close(losses["total"], 0.35 * losses["private_direct"])
    losses["total"].backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in decoder.parameters()
    )
