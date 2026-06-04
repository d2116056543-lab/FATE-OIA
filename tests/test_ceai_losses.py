import torch

from fate_oia.losses.ceai_losses import ceai_main_loss, ceai_regularizer_losses, compute_total_loss_with_gradient_budget
from fate_oia.losses.pcgrad_lite import pcgrad_project


def test_main_loss_only_final_logits_and_budget():
    outputs = {
        "final_action_logits": torch.randn(2, 4, requires_grad=True),
        "final_reason_logits": torch.randn(2, 21, requires_grad=True),
        "base_action_logits": torch.randn(2, 4, requires_grad=True),
        "base_reason_logits": torch.randn(2, 21, requires_grad=True),
        "action_specialist_logits": torch.randn(2, 4, requires_grad=True),
        "reason_specialist_logits": torch.randn(2, 21, requires_grad=True),
        "action_set_logits": torch.randn(2, 4, requires_grad=True),
        "scene_state_logits": torch.randn(2, 5, requires_grad=True),
        "pair_support": torch.randn(2, 4, 21, requires_grad=True),
        "pair_reliability": torch.sigmoid(torch.randn(2, 4, 21)),
        "pair_attention_entropy": torch.ones(2, 4, 6),
    }
    labels = {"action": torch.randint(0, 2, (2, 4)).float(), "reason": torch.randint(0, 2, (2, 21)).float()}
    main = ceai_main_loss(outputs, labels)
    assert set(main.keys()) == {"main_loss", "action_main_loss", "reason_main_loss"}
    regs = ceai_regularizer_losses(outputs, labels, {"target": torch.ones(2, 5), "mask": torch.ones(2, 5)})
    total, stats = compute_total_loss_with_gradient_budget(main, regs, gradient_budget_rho=0.15)
    assert total.requires_grad
    assert stats["gradient_budget_rho"] == 0.15
    g1 = torch.tensor([1.0, 0.0])
    g2 = torch.tensor([-1.0, 0.0])
    p1, p2, conflict = pcgrad_project(g1, g2)
    assert conflict
    assert torch.dot(p1, p2) >= -1e-6
