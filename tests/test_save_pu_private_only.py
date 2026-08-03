import pytest
import torch
from torch import nn

from fate_oia.losses.save_pu_losses import (
    admit_pu_from_train_audit,
    private_pu_loss,
    pu_score,
)


def test_pu_admission_is_continuous_and_train_audit_only() -> None:
    torch.manual_seed(3)
    targets = torch.zeros(40, 2)
    targets[:20, 0] = 1.0
    targets[:16, 1] = 1.0
    scores = torch.rand(40, 2, generator=torch.Generator().manual_seed(23)) * 0.4 + 0.3
    scores[:20, 0] += 0.02
    scores[:16, 1] = 0.90
    scores[16:, 1] = 0.10

    report = admit_pu_from_train_audit(
        scores,
        targets,
        split_name="train_audit",
        hidden_fraction=0.25,
        bootstrap_samples=128,
        seed=19,
    )

    assert report["source_split"] == "train_audit"
    assert len(report["lambda"]) == 2
    assert 0.0 < report["lambda"][0] < 0.10
    assert report["lambda"][1] == 0.10
    for row in report["labels"]:
        expected = min(0.10, max(0.0, row["bootstrap_lcb95"] / 0.05))
        assert row["lambda"] == pytest.approx(expected)
    assert all("bootstrap_lcb95" in row for row in report["labels"])

    with pytest.raises(ValueError, match="train_audit"):
        admit_pu_from_train_audit(scores, targets, split_name="test")


def test_pu_score_detaches_clean_state_and_reliability_inputs() -> None:
    clean = torch.randn(3, 2, requires_grad=True)
    state = torch.rand(3, 2, requires_grad=True)
    reliability = torch.rand(3, 2, requires_grad=True)

    score = pu_score(clean, state, reliability)

    assert not score.requires_grad
    torch.testing.assert_close(
        score,
        (torch.sigmoid(clean.detach()) * state.detach() * reliability.detach()).clamp(0.0, 1.0),
    )


def test_pu_loss_backpropagates_to_private_reason_only() -> None:
    private_head = nn.Linear(4, 2)
    private_input = torch.randn(5, 4)
    private_logits = private_head(private_input)
    action = nn.Parameter(torch.randn(2))
    clean = nn.Parameter(torch.randn(2))
    foundation = nn.Parameter(torch.randn(2))
    predicate = nn.Parameter(torch.randn(2))

    score = pu_score(clean.view(1, -1), torch.sigmoid(predicate).view(1, -1), torch.sigmoid(foundation).view(1, -1))
    loss = private_pu_loss(
        private_logits,
        torch.ones_like(private_logits),
        score.expand_as(private_logits),
        torch.tensor([0.10, 0.05]),
    )
    loss.backward()

    assert float(sum(p.grad.abs().sum() for p in private_head.parameters())) > 0.0
    assert action.grad is None
    assert clean.grad is None
    assert foundation.grad is None
    assert predicate.grad is None
