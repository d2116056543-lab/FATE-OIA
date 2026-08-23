from types import SimpleNamespace

import torch
from torch import nn

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_oia_model import TIDAOIAModel


class _ImageBase(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.foundation = nn.Module()
        self.foundation.dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=dim)
        self.foundation.predicate_head = SimpleNamespace(names=[f"p{i}" for i in range(32)])

    def encode_images(self, images):
        return self.foundation.dino(images)

    def decode_from_field(self, field, **kwargs):
        b, _, _, d = field["patch_tokens_by_layer"].shape
        action_logits = torch.randn(b, 4, device=field["patch_tokens_by_layer"].device)
        reason_logits = torch.randn(b, 21, device=field["patch_tokens_by_layer"].device)
        return {
            **field,
            "action_nodes_primary": torch.randn(b, 4, d, device=field["patch_tokens_by_layer"].device),
            "reason_nodes_primary": torch.randn(b, 21, d, device=field["patch_tokens_by_layer"].device),
            "predicate_tokens": torch.randn(b, 32, d, device=field["patch_tokens_by_layer"].device),
            "predicate_attention": torch.softmax(torch.randn(b, 32, 3600, device=field["patch_tokens_by_layer"].device), -1),
            "action_logits_primary": action_logits,
            "reason_logits_primary": reason_logits,
            "action_logits_final": action_logits,
            "reason_logits_final": reason_logits,
            "cls_tokens_by_layer": field["cls_tokens_by_layer"],
        }


def test_full_model_returns_formal_shapes_and_zero_scale_fallback():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(_ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7).eval()
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=0.0, temporal_reason_scale=0.0,
    )
    assert out["video_action_logits"].shape == (1, 4)
    assert out["video_reason_logits"].shape == (1, 21)
    assert out["history_query_tokens"].shape == (1, 14, 36, 8)
    assert torch.equal(out["action_temporal_route"], out["action_route"])
    assert torch.equal(out["video_action_logits"], out["image_action_logits"])
    assert torch.equal(out["video_reason_logits"], out["image_reason_logits"])


def test_full_model_surfaces_conditional_temporal_utility_diagnostics():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        conditional_temporal_utility=True,
        action_temporal_budget_cap=0.60,
        reason_temporal_budget_cap=0.50,
    ).eval()
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["transition_tokens_by_scale"].shape == (1, 32, 4, 8)
    assert out["action_temporal_budget"].shape == (1, 4)
    assert out["reason_temporal_budget"].shape == (1, 21)
    assert out["action_temporal_budget"].max() <= 0.60 + 1e-7
    assert out["reason_temporal_budget"].max() <= 0.50 + 1e-7
    assert torch.equal(out["video_action_logits"], out["image_action_logits"] + out["action_temporal_delta"])
    assert torch.equal(out["video_reason_logits"], out["image_reason_logits"] + out["reason_temporal_delta"])


def test_geometric_flow_is_in_final_logits_and_keeps_owner_firewall():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        geometric_flow_enabled=True, geometric_flow_hidden_dim=32,
    ).eval()
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["geometric_flow_field"].shape == (1, 13, 2, 45, 80)
    assert out["prefix_video_action_logits"].shape == (1, 4, 4)
    assert out["prefix_video_reason_logits"].shape == (1, 4, 21)
    torch.testing.assert_close(
        out["video_action_logits"],
        out["image_action_logits"] + out["semantic_action_temporal_delta"] + out["geometric_action_delta"],
    )
    reason_grads = torch.autograd.grad(
        out["video_reason_logits"].sum(), list(model.geometric_heads.action_parameters()), allow_unused=True
    )
    assert all(value is None for value in reason_grads)


def test_geometric_raw_branch_learns_while_deployment_scale_is_zero():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        geometric_flow_enabled=True, geometric_flow_hidden_dim=32,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=0.0, temporal_reason_scale=0.0,
    )
    assert torch.equal(out["video_action_logits"], out["image_action_logits"])
    loss = out["geometric_video_action_logits_raw"].sum() + out["geometric_reason_delta_raw"].sum()
    loss.backward()
    assert model.geometric_heads.action_output.weight.grad.abs().sum() > 0
    assert model.geometric_heads.reason_output.weight.grad.abs().sum() > 0
