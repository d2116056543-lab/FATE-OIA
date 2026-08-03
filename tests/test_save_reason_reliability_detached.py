import pytest
import torch

try:
    from fate_oia.models.save_reason_decoder import SAVEPrivateReasonDecoder
except ImportError:
    SAVEPrivateReasonDecoder = None


def test_private_reason_reliability_is_stop_gradient_at_all_boundaries() -> None:
    if SAVEPrivateReasonDecoder is None:
        pytest.fail("SAVEPrivateReasonDecoder is not implemented")

    torch.manual_seed(21)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    reliability = torch.rand(1, 3, requires_grad=True)
    output = decoder(
        reason_logits_clean=torch.randn(1, 3),
        global_field=torch.randn(1, 13, 8),
        detail_field=torch.randn(1, 13, 8),
        factor_measurement_token=torch.randn(1, 3, 8),
        factor_evidence_map=torch.rand(1, 3, 13),
        factor_reliability=reliability,
        progress=0.73,
    )
    output["reason_logits_benchmark"].square().mean().backward()

    assert not output["reason_reliability"].requires_grad
    assert reliability.grad is None or torch.count_nonzero(reliability.grad) == 0


def _private_inputs(clean: torch.Tensor, reliability: torch.Tensor) -> dict[str, torch.Tensor]:
    batch, reasons = clean.shape
    return {
        "reason_logits_clean": clean,
        "global_field": torch.randn(batch, 11, 8),
        "detail_field": torch.randn(batch, 11, 8),
        "factor_measurement_token": torch.randn(batch, reasons, 8),
        "factor_evidence_map": torch.rand(batch, reasons, 11),
        "factor_reliability": reliability,
    }


def test_private_kappa_is_exact_clamped_fraction_of_clean_rms_and_bounds_delta() -> None:
    torch.manual_seed(22)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    clean = torch.tensor([[0.10, 1.0, 10.0], [-0.10, -1.0, -10.0]])
    output = decoder(
        **_private_inputs(clean, torch.zeros_like(clean)),
        progress=1.0,
        update_running_stats=True,
    )
    expected_rms = clean.square().mean(0).sqrt()
    expected_kappa = (0.35 * expected_rms).clamp(0.20, 2.00)

    torch.testing.assert_close(output["reason_clean_logit_rms_ema"], expected_rms)
    torch.testing.assert_close(output["reason_private_kappa"], expected_kappa)
    assert torch.all(
        output["reason_private_delta_bounded"].abs()
        <= output["reason_private_kappa"].view(1, -1) + 1e-7
    )


@pytest.mark.parametrize(
    ("progress", "expected_ramp"),
    [(-1.0, 0.0), (0.0, 0.0), (0.05, 0.5), (0.10, 1.0), (4.0, 1.0)],
)
def test_private_reason_ramp_saturates_exactly(progress: float, expected_ramp: float) -> None:
    torch.manual_seed(23)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    clean = torch.randn(1, 3)
    output = decoder(
        **_private_inputs(clean, torch.zeros_like(clean)),
        progress=progress,
    )

    torch.testing.assert_close(
        output["reason_benchmark_ramp"],
        clean.new_tensor(expected_ramp),
        atol=0.0,
        rtol=0.0,
    )


def test_private_benchmark_applies_detached_reliability_once() -> None:
    torch.manual_seed(24)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    clean = torch.randn(1, 3)
    reliability = torch.tensor([[0.0, 0.5, 1.0]], requires_grad=True)
    output = decoder(
        **_private_inputs(clean, reliability),
        progress=0.05,
    )
    expected = clean + 0.5 * (1.0 - reliability.detach()) * output[
        "reason_private_delta_bounded"
    ]

    torch.testing.assert_close(output["reason_logits_benchmark"], expected)
    assert not output["reason_reliability"].requires_grad
