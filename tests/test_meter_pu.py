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


def test_pu_zero_lambda_is_exactly_inactive() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.randint(0, 2, (4, 3)).float()
    score = torch.rand(4, 3)
    loss = meter_private_pu_loss(logits, targets, score, torch.zeros(3))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_pu_lambda_only_updates_admitted_labels() -> None:
    logits = torch.zeros(2, 3, requires_grad=True)
    targets = torch.ones(2, 3)
    score = torch.ones(2, 3)
    loss = meter_private_pu_loss(logits, targets, score, torch.tensor([0.15, 0.0, 0.15]))
    loss.backward()
    assert float(logits.grad[:, 1].abs().sum()) == 0.0
    assert float(logits.grad[:, 0].abs().sum()) > 0.0
    assert float(logits.grad[:, 2].abs().sum()) > 0.0
