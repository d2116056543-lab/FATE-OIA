from types import SimpleNamespace

import torch

from fate_oia.losses.tida_loss_registry import assert_owner_exact_cover
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


class _DummyObjectTracker(nn.Module):
    def forward(self, video):
        batch, frames = video.shape[:2]
        x = torch.linspace(-0.8, 0.8, 9, device=video.device)
        time = torch.linspace(0.0, 1.0, frames, device=video.device)
        xy = torch.zeros(batch, frames, 9, 2, device=video.device)
        xy[..., 0] = x[None, None] + 0.20 * time[None, :, None] * x[None, None]
        xy[..., 1] = (
            torch.linspace(-0.2, 0.4, frames, device=video.device)[None, :, None]
            + 0.10 * time[None, :, None].square() * x[None, None]
        )
        return {
            "object_tracks_xy": xy,
            "object_tracks_visibility": torch.ones(
                batch, frames, 9, dtype=torch.bool, device=video.device
            ),
            "object_tracks_visibility_rate": torch.ones(batch, device=video.device),
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


def test_object_intent_transport_reaches_final_logits_with_task_firewall():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        object_intent_enabled=True, object_tracker=_DummyObjectTracker(),
        object_intent_action_cap=0.08, object_intent_reason_cap=0.06,
    )
    with torch.no_grad():
        model.object_intent.action_output.weight.fill_(0.05)
        model.object_intent.reason_output.weight.fill_(0.05)
    output = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )

    torch.testing.assert_close(
        output["video_action_logits"],
        output["pre_object_intent_video_action_logits"]
        + output["object_intent_action_delta_scaled"],
    )
    torch.testing.assert_close(
        output["video_reason_logits"],
        output["pre_object_intent_video_reason_logits"]
        + output["object_intent_reason_delta_scaled"],
    )
    assert output["object_tracks_xy"].shape == (1, 15, 9, 2)
    assert "object_intent_action" in model.owner_parameters()
    assert "object_intent_reason" in model.owner_parameters()
    action_parameters = list(model.object_intent.action_encoder.parameters()) + list(
        model.object_intent.action_output.parameters()
    )
    gradients = torch.autograd.grad(
        output["video_reason_logits"].sum(), action_parameters, allow_unused=True
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)
    reverse = model.rerun_temporal_from_output(
        output, "time_reverse", temporal_action_scale=1.0, temporal_reason_scale=1.0
    )
    assert "object_intent_action_delta_scaled" in reverse
    assert not torch.allclose(
        output["object_intent_motion_features"],
        reverse["object_intent_motion_features"],
    )


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
        geometric_flow_enabled=True, geometric_flow_hidden_dim=64,
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
        geometric_flow_enabled=True, geometric_flow_hidden_dim=64,
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


def test_traffic_action_motion_is_in_final_action_and_not_reason():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7,
        traffic_action_enabled=True,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["traffic_action_delta"].shape == (1, 4)
    assert out["traffic_action_attention"].shape[:2] == (1, 4)
    assert torch.allclose(
        out["video_action_logits"],
        out["image_action_logits"] + out["semantic_action_temporal_delta"]
        + out["geometric_action_delta"] + out["traffic_action_delta"],
    )
    gradient = torch.autograd.grad(
        out["video_reason_logits"].sum(), list(model.traffic_action.parameters()), allow_unused=True
    )
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in gradient)


def test_trajectory_relational_traffic_is_terminal_anchored_and_in_final_action_only():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7, traffic_motion_topk=4,
        traffic_trajectory_enabled=True, traffic_trajectory_cap=0.08,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["trajectory_xy"].shape == (1, 4, 4, 15, 2)
    assert out["trajectory_local_candidate_coverage"].shape == (1, 4, 4, 15)
    assert out["trajectory_interaction_risk"].shape == (1, 4, 4)
    assert out["trajectory_attention"].shape == (1, 4, 4)
    assert out["terminal_action_patch_xy"].shape == (1, 4, 4, 2)
    torch.testing.assert_close(out["trajectory_xy"][..., -1, :], out["terminal_action_patch_xy"])
    torch.testing.assert_close(
        out["video_action_logits"],
        out["image_action_logits"] + out["semantic_action_temporal_delta"]
        + out["geometric_action_delta"] + out["traffic_action_delta"]
        + out["traffic_trajectory_delta"],
    )
    assert torch.count_nonzero(out["traffic_trajectory_delta"]) == 0
    assert "traffic_trajectory" in model.owner_parameters()
    gradient = torch.autograd.grad(
        out["video_reason_logits"].sum(), list(model.traffic_trajectory_head.parameters()), allow_unused=True
    )
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in gradient)


def test_semantic_relational_traffic_reaches_action_and_reason_with_branch_firewall():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7, traffic_motion_topk=4,
        relational_traffic_enabled=True,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["semantic_trajectory_xy"].shape == (1, 1, 4, 15, 2)
    assert out["relational_action_attention"].shape == (1, 4, 4)
    assert out["relational_reason_attention"].shape == (1, 21, 4)
    assert torch.count_nonzero(out["relational_action_delta"]) == 0
    assert torch.count_nonzero(out["relational_reason_delta"]) == 0
    assert "relational_traffic_action" in model.owner_parameters()
    assert "relational_traffic_reason" in model.owner_parameters()
    assert_owner_exact_cover(model, model.owner_parameters())
    action_parameters = model.owner_parameters()["relational_traffic_action"]
    reason_parameters = model.owner_parameters()["relational_traffic_reason"]
    action_from_reason = torch.autograd.grad(
        out["video_reason_logits"].sum(), action_parameters, allow_unused=True, retain_graph=True
    )
    reason_from_action = torch.autograd.grad(
        out["video_action_logits"].sum(), reason_parameters, allow_unused=True
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in action_from_reason)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in reason_from_action)
