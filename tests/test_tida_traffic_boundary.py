import torch

from fate_oia.models.tida_traffic_boundary import TIDATrafficAdaptiveBoundary


def _inputs(batch=2):
    return (
        torch.randn(batch, 4, requires_grad=True),
        torch.randn(batch, 4, requires_grad=True),
        torch.randn(batch, 4, 8, requires_grad=True),
        torch.rand(batch, 4, requires_grad=True),
        torch.rand(batch, 4, requires_grad=True),
        torch.rand(batch, 4, 3, requires_grad=True),
    )


def test_traffic_boundary_is_exact_zero_effect_and_trainable_at_initialization():
    head = TIDATrafficAdaptiveBoundary()
    inputs = _inputs()
    output = head(*inputs)
    torch.testing.assert_close(output["traffic_adaptive_deploy_action_logits"], inputs[0])
    output["traffic_adaptive_deploy_action_logits"].sum().backward()
    assert head.network[-1].weight.grad is not None
    assert head.network[-1].weight.grad.abs().sum() > 0


def test_traffic_boundary_detaches_all_evidence_but_not_the_base_logit_path():
    head = TIDATrafficAdaptiveBoundary()
    with torch.no_grad():
        head.network[-1].weight.fill_(0.1)
    inputs = _inputs()
    output = head(*inputs)
    output["traffic_adaptive_boundary_delta"].sum().backward()
    assert inputs[0].grad is None
    assert all(value.grad is None for value in inputs[1:])


def test_traffic_boundary_is_action_specific_and_bounded():
    head = TIDATrafficAdaptiveBoundary(cap=0.2)
    with torch.no_grad():
        head.network[-1].weight.fill_(2.0)
    inputs = list(_inputs(batch=1))
    inputs[2] = inputs[2].expand(1, 4, 8).clone()
    output = head(*inputs)
    assert torch.all(output["traffic_adaptive_boundary_delta"].abs() <= 0.2 + 1e-7)
    assert output["traffic_adaptive_boundary_delta"].std() > 0
