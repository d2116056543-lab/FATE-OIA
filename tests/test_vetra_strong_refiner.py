import torch

from fate_oia.models.vetra_strong_refiner import SelectiveVisualActionRankRefiner


def test_zero_init_is_exact_identity_and_reason_is_untouched():
    module = SelectiveVisualActionRankRefiner(dim=16, rank=8, action_dim=4, max_delta=0.12)
    base_action = torch.randn(3, 4)
    reason = torch.randn(3, 21)
    nodes = torch.randn(3, 4, 16)
    evidence = torch.randn(3, 4, 2, 16)
    output = module(base_action, reason, nodes, evidence)
    torch.testing.assert_close(output["action_logits_final"], base_action, rtol=0, atol=0)
    assert output["reason_logits_final"].data_ptr() == reason.data_ptr()


def test_delta_is_bounded_and_gain_can_disable_each_action():
    module = SelectiveVisualActionRankRefiner(dim=16, rank=8, action_dim=4, max_delta=0.12)
    torch.nn.init.normal_(module.output_weight, std=2.0)
    module.set_deployment_gain(torch.tensor([1.0, 0.0, 0.5, 1.0]))
    output = module(
        torch.zeros(5, 4),
        torch.zeros(5, 21),
        torch.randn(5, 4, 16),
        torch.randn(5, 4, 3, 16),
    )
    assert output["action_delta"].abs().max() <= 0.120001
    assert torch.count_nonzero(output["action_delta"][:, 1]) == 0


def test_backward_updates_refiner_without_touching_detached_inputs():
    module = SelectiveVisualActionRankRefiner(dim=16, rank=8, action_dim=4, max_delta=0.12)
    nodes = torch.randn(6, 4, 16, requires_grad=True)
    evidence = torch.randn(6, 4, 2, 16, requires_grad=True)
    output = module(torch.randn(6, 4), torch.randn(6, 21), nodes, evidence)
    output["action_logits_final"].sum().backward()
    assert module.output_weight.grad is not None
    assert torch.isfinite(module.output_weight.grad).all()
    assert nodes.grad is None
    assert evidence.grad is None
