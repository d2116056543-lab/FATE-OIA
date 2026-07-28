import torch

from fate_oia.losses.meter_pu_losses import meter_hidden_positive_audit, meter_private_pu_loss


def test_pu_zero_lambda_is_finite_and_audit_is_data_driven() -> None:
    logits = torch.randn(6, 21, requires_grad=True)
    targets = torch.randint(0, 2, (6, 21)).float()
    score = torch.rand(6, 21)
    loss = meter_private_pu_loss(logits, targets, score, torch.zeros(21))
    loss.backward()
    report = meter_hidden_positive_audit(torch.sigmoid(logits.detach()), score, targets, min_positive_count=1)
    assert torch.isfinite(loss)
    assert "labels" in report and "active_labels" in report
