import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.losses.asymmetric_loss import AsymmetricLossMultiLabel
from fate_oia.models.snna25_head import SNNA25Head


def test_snna25_head_splits_action_and_reason_logits():
    head = SNNA25Head(16, action_dim=4, reason_dim=21)
    out = head(torch.randn(3, 16))
    assert out["action_logits"].shape == (3, 4)
    assert out["reason_logits"].shape == (3, 21)
    assert out["logits"].shape == (3, 25)


def test_asymmetric_loss_is_multilabel_and_finite():
    loss_fn = AsymmetricLossMultiLabel()
    logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_eval_snna25_reports_action_and_reason_metrics():
    logits = torch.randn(5, 25)
    labels = torch.zeros(5, 25)
    labels[:, 0] = 1
    labels[:, 4] = 1
    result = evaluate_snna25(logits, labels, action_dim=4, threshold_mode="fixed")
    assert result["action_dim"] == 4
    assert result["reason_dim"] == 21
    assert "Act_mF1" in result["metrics"]
    assert "Exp_mF1" in result["metrics"]