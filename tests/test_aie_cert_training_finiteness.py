import pytest
import torch

from fate_oia.engine.train_aie_cert_oia import (
    assert_finite_gradients,
    assert_finite_loss,
    assert_finite_parameters,
)
from fate_oia.losses.aie_cert_loss_registry import AIECertLossRegistry
from fate_oia.losses.aie_losses import predicate_map_compactness_loss, predicate_map_loss


def test_sparse_predicate_losses_have_finite_gradients():
    logits = torch.randn(2, 3, 3600, requires_grad=True)
    attention = torch.softmax(logits, dim=-1)
    attention = attention * (attention > attention.quantile(0.8, dim=-1, keepdim=True))
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    target = torch.zeros_like(attention)
    target[..., 120:180] = 1.0
    mask = torch.ones(2, 3)

    loss = predicate_map_loss(attention, target, mask) + predicate_map_compactness_loss(attention)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_training_finiteness_guards_reject_nonfinite_values():
    model = torch.nn.Linear(2, 1)
    registry = AIECertLossRegistry()
    registry.add("bad", "owner", torch.tensor(float("nan")), 1.0)
    with pytest.raises(FloatingPointError, match="terms=\\['bad'\\]"):
        assert_finite_loss(torch.tensor(float("nan")), registry, 4)

    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        assert_finite_gradients(model, 4)

    with torch.no_grad():
        model.bias.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite parameters"):
        assert_finite_parameters(model, 4)
