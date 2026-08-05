import torch


def _gradient_for_conflict(conflict_value: float) -> torch.Tensor:
    from fate_oia.losses.lens_latent_losses import conflict_safe_reason_logits

    state_logits = torch.randn(2, 21, 3, requires_grad=True)
    state_prob = state_logits.softmax(-1)
    source_reason = torch.randn(2, 21, requires_grad=True)
    emission = torch.tensor([[0.05, 0.4, 0.95]]).expand(21, -1)
    observed = torch.zeros(2, 21)
    gamma = torch.zeros_like(state_prob)
    gamma[..., 0] = 0.45
    gamma[..., 1] = 0.45
    gamma[..., 2] = 0.10
    result = conflict_safe_reason_logits(
        state_prob, source_reason, emission, observed, gamma,
        torch.full((2, 21), conflict_value), alpha_reason=0.7,
    )
    result["reason_logits_formal_train"].sum().backward()
    return state_logits.grad.norm() + source_reason.grad.norm()


def test_conflict_discount_changes_shared_gradient_not_forward_values():
    from fate_oia.losses.lens_latent_losses import conflict_safe_reason_logits

    state = torch.softmax(torch.randn(2, 21, 3), -1).requires_grad_()
    source = torch.randn(2, 21, requires_grad=True)
    emission = torch.tensor([[0.05, 0.4, 0.95]]).expand(21, -1)
    observed = torch.zeros(2, 21)
    gamma = torch.softmax(torch.randn(2, 21, 3), -1)
    low = conflict_safe_reason_logits(state, source, emission, observed, gamma, torch.zeros(2, 21), 0.7)
    high = conflict_safe_reason_logits(state, source, emission, observed, gamma, torch.ones(2, 21), 0.7)
    assert torch.allclose(low["reason_logits_formal_train"], high["reason_logits_formal_train"])
    assert _gradient_for_conflict(0.95) < _gradient_for_conflict(0.0)
    assert float(high["share_weight"].min()) >= 0.05

