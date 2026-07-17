import torch

from fate_oia.models.mosaic_reason_policy import bounded_reason_residual


def test_reason_residual_is_bounded_and_zero_safe():
    visual = torch.randn(2, 21)
    annotation = torch.randn(2, 21)
    final, alpha = bounded_reason_residual(visual, annotation, init_alpha=0.05, max_alpha=0.25)
    assert torch.isfinite(final).all()
    assert float(alpha.max()) <= 0.25
    assert torch.max(torch.abs(final - visual)) <= 0.25 * torch.max(torch.abs(annotation - visual)) + 1e-6

