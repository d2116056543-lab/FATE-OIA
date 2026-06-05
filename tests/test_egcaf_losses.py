import torch
from fate_oia.losses.egcaf_losses import EGCafLoss
from fate_oia.losses.egcaf_gradient_budget import true_gradient_budget


def test_loss_has_core_final_reason_and_true_budget():
    p = torch.nn.Parameter(torch.tensor([1.0]))
    main = (p ** 2).sum()
    aux = ((p + 1) ** 2).sum()
    scaled, stats = true_gradient_budget(main, aux, [p])
    assert stats["used_true_grad_norm"] is True
    assert stats["norm_main"] > 0 and stats["norm_aux"] > 0
    loss = EGCafLoss()
    out = {
        "action_core_logits": torch.randn(2,4, requires_grad=True),
        "action_final_logits": torch.randn(2,4, requires_grad=True),
        "reason_logits": torch.randn(2,21, requires_grad=True),
        "factor_weights": torch.softmax(torch.randn(2,4,12), -1),
        "selected_weights": torch.softmax(torch.randn(2,4,3), -1),
        "z_selected_only": torch.randn(2,4, requires_grad=True),
        "z_without_selected": torch.randn(2,4, requires_grad=True),
        "z_without_random": torch.randn(2,4, requires_grad=True),
    }
    total, stats2 = loss(out, torch.rand(2,4).round(), torch.rand(2,21).round())
    assert total.requires_grad
    assert {"loss_action_core", "loss_action_final", "loss_reason"}.issubset(stats2)
