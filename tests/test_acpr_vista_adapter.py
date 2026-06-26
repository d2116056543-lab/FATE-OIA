from __future__ import annotations

import torch

from fate_oia.models.acpr_visual_token_adapter import ACPRPredicateAnchoredVisualAdapter


def test_vista_adapter_zero_gate_preserves_tokens_and_shapes():
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=384, rank=48, num_layers=3, num_predicates=32)
    x = torch.randn(2, 3, 3600, 384)
    probs = torch.rand(2, 32)
    attn = torch.softmax(torch.randn(2, 32, 3600), dim=-1)
    y, stats = adapter(x, probs, attn, epoch=0)
    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-6)
    assert stats["vista_gate_map"].shape == (2, 3600)
    assert float(stats["vista_gate_mean"]) >= 0.20


def test_vista_adapter_uses_zero_up_nonzero_scale_startup():
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=64, rank=8, num_layers=3, num_predicates=8, grid_hw=(8, 8))
    for block in adapter.blocks:
        assert torch.allclose(block.up.weight, torch.zeros_like(block.up.weight))
        assert torch.allclose(block.up.bias, torch.zeros_like(block.up.bias))

    x = torch.randn(1, 3, 3600, 384)
    x = torch.randn(1, 3, 64, 64)
    probs = torch.rand(1, 8)
    attn = torch.softmax(torch.randn(1, 8, 64), dim=-1)
    y, _ = adapter(x, probs, attn, epoch=0)
    target = x + 0.05 * torch.tanh(x.roll(shifts=1, dims=2))
    assert torch.allclose(y, x, atol=1e-6)
    loss = (y - target).square().mean()
    loss.backward()
    up_grad_first = sum(float(block.up.weight.grad.abs().sum()) for block in adapter.blocks)
    down_grad_first = sum(float(block.down.weight.grad.abs().sum()) for block in adapter.blocks if block.down.weight.grad is not None)
    assert up_grad_first > 0
    assert down_grad_first == 0.0


def test_vista_adapter_second_backward_reaches_internal_layers():
    adapter = ACPRPredicateAnchoredVisualAdapter(dim=64, rank=8, num_layers=3, num_predicates=8, grid_hw=(8, 8))
    opt = torch.optim.SGD(adapter.parameters(), lr=1.0)
    x = torch.randn(1, 3, 64, 64)
    probs = torch.rand(1, 8)
    attn = torch.softmax(torch.randn(1, 8, 64), dim=-1)
    target = x + 0.05 * torch.tanh(x.roll(shifts=1, dims=2))
    y, _ = adapter(x, probs, attn, epoch=0)
    ((y - target).square().mean()).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    y2, _ = adapter(x, probs, attn, epoch=0)
    ((y2 - target).square().mean()).backward()
    down_grad = sum(float(block.down.weight.grad.abs().sum()) for block in adapter.blocks)
    depthwise_grad = sum(float(block.depthwise.weight.grad.abs().sum()) for block in adapter.blocks)
    up_grad = sum(float(block.up.weight.grad.abs().sum()) for block in adapter.blocks)
    assert down_grad > 0
    assert depthwise_grad > 0
    assert up_grad > 0


def test_vista_predicate_gate_has_annealing_and_importance_prior():
    adapter = ACPRPredicateAnchoredVisualAdapter(
        dim=64,
        rank=8,
        num_layers=3,
        num_predicates=4,
        grid_hw=(8, 8),
        predicate_names=[
            "front_vehicle_close",
            "global_scene_context",
            "traffic_light_green",
            "drivable_center",
        ],
        reliable_predicate_weight=1.0,
        global_predicate_weight=0.3,
        unreliable_predicate_weight=0.0,
        anchor_mix_start_epoch=2,
        anchor_mix_end_epoch=5,
        early_global_gate=True,
    )
    prior = adapter.predicate_importance_prior.detach()
    assert prior[0] > prior[1] > prior[2]
    assert prior[3] > prior[1]
    x = torch.randn(1, 3, 64, 64)
    probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    attn = torch.zeros(1, 4, 64)
    attn[:, 0, :8] = 1.0 / 8.0
    attn[:, 1:, :] = 1.0 / 64.0
    _, early = adapter(x, probs, attn, epoch=0)
    _, late = adapter(x, probs, attn, epoch=6)
    assert float(early["vista_anchor_mix"]) == 0.0
    assert float(late["vista_anchor_mix"]) == 1.0
    assert float(early["vista_gate_map"].std()) == 0.0
    assert float(late["vista_gate_map"].std()) > 0.0
