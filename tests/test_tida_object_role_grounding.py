import torch

from fate_oia.datasets.bdd100k_object_roles import annotation_to_patch_roles
from fate_oia.models.tida_object_intent_flow import TIDAObjectRoleHead, TIDAObjectIntentTransport


def test_annotation_to_patch_roles_maps_boxes_and_drivable_polygon():
    annotation = {"frames": [{"objects": [
        {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 640, "y2": 360}},
        {"category": "area/drivable", "poly2d": [
            {"value": [640, 360, "L"]}, {"value": [1280, 360, "L"]},
            {"value": [1280, 720, "L"]}, {"value": [640, 720, "L"]},
        ]},
    ]}]}
    roles = annotation_to_patch_roles(annotation, source_hw=(720, 1280), grid_hw=(45, 80))
    assert roles.shape == (45, 80)
    assert (roles[:23, :40] == 1).float().mean() > 0.95
    assert (roles[23:, 40:] == 4).float().mean() > 0.90


def test_role_head_is_a_visual_probability_distribution():
    head = TIDAObjectRoleHead(dim=16, num_roles=5)
    logits = head(torch.randn(2, 7, 16))
    assert logits.shape == (2, 7, 5)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.softmax(-1).sum(-1), torch.ones(2, 7), atol=1e-6)


def test_object_transport_uses_predicted_roles_and_exports_role_mass():
    transport = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=3, heads=4)
    tracks = torch.zeros(2, 5, 4, 2)
    tracks[:, :, :, 0] = torch.linspace(-0.4, 0.4, 5)[None, :, None]
    visibility = torch.ones(2, 5, 4, dtype=torch.bool)
    patches = torch.randn(2, 16, 16)
    action_nodes = torch.randn(2, 4, 16)
    reason_nodes = torch.randn(2, 3, 16)
    out = transport(tracks, visibility, patches, (4, 4), action_nodes, reason_nodes)
    assert out["object_intent_track_role_probs"].shape == (2, 4, 5)
    assert out["object_intent_action_role_mass"].shape == (2, 4, 5)
    assert out["object_intent_reason_role_mass"].shape == (2, 3, 5)
    assert torch.allclose(out["object_intent_action_role_mass"].sum(-1), torch.ones(2, 4), atol=1e-5)


def test_frozen_role_head_blocks_action_and_reason_gradients():
    transport = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=3, heads=4)
    transport.freeze_role_head()
    tracks = torch.randn(2, 5, 4, 2).clamp(-1, 1)
    visibility = torch.ones(2, 5, 4, dtype=torch.bool)
    patches = torch.randn(2, 16, 16)
    out = transport(tracks, visibility, patches, (4, 4), torch.randn(2, 4, 16), torch.randn(2, 3, 16))
    (out["object_intent_action_candidate"].sum() + out["object_intent_reason_candidate"].sum()).backward()
    assert all(parameter.grad is None for parameter in transport.role_head.parameters())


def test_track_aligned_semantics_uses_visible_history_and_exact_terminal_field():
    transport = TIDAObjectIntentTransport(dim=4, num_actions=2, num_reasons=2, heads=2)
    dense = torch.zeros(1, 3, 4, 4)
    dense[:, 0, :, 0] = 2.0
    dense[:, 1, :, 1] = 4.0
    dense[:, 2, :, 2] = 8.0
    terminal = torch.zeros(1, 16, 4)
    terminal[:, :, 3] = 16.0
    tracks = torch.zeros(1, 3, 1, 2)
    visible = torch.ones(1, 3, 1, dtype=torch.bool)

    temporal, pooled, weights = transport.sample_track_aligned_semantics(
        dense, (2, 2), tracks, visible,
        terminal_patch_tokens=terminal, terminal_grid_hw=(4, 4),
    )

    assert temporal.shape == (1, 3, 1, 4)
    assert temporal[0, -1, 0, 3] == 16.0
    assert pooled[0, 0, 0] > 0.0
    assert pooled[0, 0, 1] > 0.0
    assert pooled[0, 0, 3] > 0.0
    assert torch.allclose(weights.sum(1), torch.ones(1, 1), atol=1e-6)


def test_track_aligned_semantics_ignores_invisible_history():
    transport = TIDAObjectIntentTransport(dim=4, num_actions=2, num_reasons=2, heads=2)
    dense_a = torch.zeros(1, 2, 4, 4)
    dense_b = dense_a.clone()
    dense_b[:, 0] = 1000.0
    tracks = torch.zeros(1, 2, 1, 2)
    visible = torch.tensor([[[False], [True]]])

    _, pooled_a, weights_a = transport.sample_track_aligned_semantics(
        dense_a, (2, 2), tracks, visible,
    )
    _, pooled_b, weights_b = transport.sample_track_aligned_semantics(
        dense_b, (2, 2), tracks, visible,
    )

    assert torch.allclose(pooled_a, pooled_b)
    assert weights_a[0, 0, 0] == 0.0
    assert torch.allclose(weights_a, weights_b)


def test_object_transport_exports_temporal_role_consistency():
    transport = TIDAObjectIntentTransport(dim=16, num_actions=4, num_reasons=3, heads=4)
    tracks = torch.randn(2, 5, 4, 2).clamp(-1, 1)
    visibility = torch.ones(2, 5, 4, dtype=torch.bool)
    temporal_patches = torch.randn(2, 5, 16, 16)
    terminal_patches = torch.randn(2, 16, 16)
    out = transport(
        tracks, visibility, terminal_patches, (4, 4),
        torch.randn(2, 4, 16), torch.randn(2, 3, 16),
        temporal_patch_tokens=temporal_patches,
        temporal_grid_hw=(4, 4),
    )
    assert out["object_intent_track_semantics_by_frame"].shape == (2, 5, 4, 16)
    assert out["object_intent_track_role_probs_by_frame"].shape == (2, 5, 4, 5)
    assert out["object_intent_semantic_temporal_weights"].shape == (2, 5, 4)
    assert out["object_intent_track_role_consistency"].shape == (2, 4)
    assert torch.isfinite(out["object_intent_track_role_consistency"]).all()
