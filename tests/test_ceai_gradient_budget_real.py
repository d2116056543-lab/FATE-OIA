import torch

from fate_oia.losses.gradient_budget import compute_gradient_budget_scale


def test_gradient_budget_uses_true_gradient_norms_not_loss_values():
    w = torch.nn.Parameter(torch.tensor([1.0]))
    main_loss = (w * 2.0).sum()
    aux_loss = (w * 10.0).sum()
    scale, stats = compute_gradient_budget_scale(main_loss, aux_loss, [w], rho=0.2)
    assert stats["used_true_grad_norm"] is True
    assert abs(stats["norm_main"] - 2.0) < 1e-5
    assert abs(stats["norm_aux"] - 10.0) < 1e-5
    assert 0.03 <= float(scale) <= 0.05
