from __future__ import annotations

import torch

from fate_oia.explain.fate_snna import gradient_x_attention, label_attention_map, smoothgrad_token_attribution, snna_value_grad


def test_label_attention_target_label_changes_heatmap():
    attention = torch.zeros(1, 2, 3, 5)
    attention[:, :, 0, 1] = 1.0
    attention[:, :, 1, 3] = 1.0
    heat0 = label_attention_map(attention, 0)
    heat1 = label_attention_map(attention, 1)
    assert not torch.allclose(heat0, heat1)


def test_snna_value_grad_uses_positive_gradient_and_value_norm():
    attention = torch.ones(1, 1, 2, 4)
    grad = torch.tensor([[[[0.0, 1.0, -1.0, 0.5], [0.0, 0.0, 1.0, 0.0]]]])
    values = torch.ones(1, 1, 4, 3)
    heat = snna_value_grad(attention, values, grad, 0)
    assert heat.shape == (1, 4)
    assert float(heat[0, 1]) > float(heat[0, 2])


def test_smoothgrad_token_attribution_runs_and_changes_with_label():
    torch.manual_seed(0)
    linear = torch.nn.Linear(6, 2)

    def forward(tokens):
        return linear(tokens.mean(1))

    tokens = torch.randn(2, 4, 6)
    heat = smoothgrad_token_attribution(forward, tokens, 0, samples=1, sigma=0.01)
    assert heat.shape == (2, 4)
    assert torch.isfinite(heat).all()
