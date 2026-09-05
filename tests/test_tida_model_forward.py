from types import SimpleNamespace

import torch

from fate_oia.losses.tida_loss_registry import assert_owner_exact_cover
from torch import nn

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_oia_model import TIDAFrozenVETRAImageBase, TIDAOIAModel
from fate_oia.engine.train_tida_oia import append_supervision_tensors


class _StageCCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("thresholds", torch.tensor([0.48, 0.355, 0.38, 0.305]))

    def forward(self, original, flipped):
        logits = original + 0.25 * flipped
        return {"action_logits": logits, "action_deploy_logits": logits}


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
            "label_attention": torch.nn.functional.one_hot(
                torch.arange(25, device=field["patch_tokens_by_layer"].device),
                num_classes=3600,
            ).float()[None].expand(b, -1, -1),
            "predicate_tokens": torch.randn(b, 32, d, device=field["patch_tokens_by_layer"].device),
            "predicate_attention": torch.softmax(torch.randn(b, 32, 3600, device=field["patch_tokens_by_layer"].device), -1),
            "action_logits_primary": action_logits,
            "reason_logits_primary": reason_logits,
            "action_logits_final": action_logits,
            "reason_logits_final": reason_logits,
            "cls_tokens_by_layer": field["cls_tokens_by_layer"],
        }

    def decode_history_from_field(self, field):
        return self.decode_from_field(field)


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


def test_frozen_vetra_history_decoder_uses_primary_head_not_full_evidence_reread():
    class Foundation(nn.Module):
        def decode_field(self, field):
            batch = field["patch_tokens_by_layer"].shape[0]
            return {
                "action_logits_primary": torch.full((batch, 4), 1.25),
                "reason_logits_primary": torch.full((batch, 21), -0.75),
            }

    class ImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.foundation = Foundation()

        def decode_from_field(self, *args, **kwargs):
            raise AssertionError("history frames must not invoke the full evidence decoder")

    wrapper = TIDAFrozenVETRAImageBase(ImageModel())
    field = {"patch_tokens_by_layer": torch.randn(2, 3, 3600, 8)}

    decoded = wrapper.decode_history_from_field(field)

    torch.testing.assert_close(decoded["action_logits_final"], torch.full((2, 4), 1.25))
    torch.testing.assert_close(decoded["reason_logits_final"], torch.full((2, 21), -0.75))


def test_frozen_vetra_stage_c_is_real_action_path_and_exposes_thresholds():
    reason_thresholds = torch.linspace(0.1, 0.9, 21)
    wrapper = TIDAFrozenVETRAImageBase(
        _ImageBase(),
        stage_c_action_calibrator=_StageCCalibrator(),
        stage_c_reason_thresholds=reason_thresholds,
    )
    original = {
        "action_logits_final": torch.ones(2, 4),
        "reason_logits_final": torch.randn(2, 21),
    }
    flipped = {
        "action_logits_final": torch.full((2, 4), 2.0),
        "reason_logits_final": torch.randn(2, 21),
    }

    deployed = wrapper.apply_stage_c(original, flipped)

    torch.testing.assert_close(deployed["action_logits_final"], torch.full((2, 4), 1.5))
    torch.testing.assert_close(deployed["reason_logits_final"], original["reason_logits_final"])
    torch.testing.assert_close(
        wrapper.stage_c_thresholds(),
        torch.cat((_StageCCalibrator().thresholds, reason_thresholds)),
    )


def test_tida_forward_uses_original_and_canonical_flip_for_stage_c_image_action():
    image_base = _ImageBase()
    wrapper = TIDAFrozenVETRAImageBase(
        image_base,
        stage_c_action_calibrator=_StageCCalibrator(),
        stage_c_reason_thresholds=torch.full((21,), 0.5),
    )
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        wrapper, dim=8, predicate_roles=roles,
        history_encoder_mode="terminal_repeat", context_chunk_size=2,
    ).eval()

    output = model(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 2, 3, 192, 344),
        torch.linspace(-1, 0, 3).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        temporal_action_scale=0.0,
        temporal_reason_scale=0.0,
    )

    expected = (
        output["image_branch"]["action_logits_pre_stage_c"]
        + 0.25 * output["image_branch"]["action_logits_flip_stage_c"]
    )
    torch.testing.assert_close(output["image_action_logits"], expected)
    torch.testing.assert_close(output["video_action_logits"], expected)


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
    assert torch.equal(
        out["reason_local_centered_candidate_logits"], out["image_reason_logits"]
    )
    assert torch.count_nonzero(out["reason_local_centered_candidate_delta"]) == 0
    assert torch.equal(out["video_action_logits"], out["image_action_logits"])
    assert torch.equal(out["video_reason_logits"], out["image_reason_logits"])
    assert out["relational_reason_soft_selected_deleted_delta"].shape == (1, 21)
    assert out["relational_reason_soft_control_deleted_delta"].shape == (1, 21)


def test_legacy_semantic_routes_can_be_disabled_without_disabling_private_traffic():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(),
        dim=8,
        predicate_roles=roles,
        context_chunk_size=2,
        traffic_trajectory_enabled=True,
        legacy_semantic_routes_enabled=False,
    ).eval()

    out = model(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 2, 3, 192, 344),
        torch.linspace(-1, 0, 3).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        temporal_action_scale=1.0,
        temporal_reason_scale=1.0,
    )

    assert torch.count_nonzero(out["legacy_semantic_action_temporal_delta"]) == 0
    assert torch.count_nonzero(out["reason_temporal_delta_raw"]) == 0
    assert out["traffic_trajectory_candidate_delta"].shape == (1, 4)


def test_logit_flow_reuses_history_image_head_and_preserves_task_firewall():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=2,
        logit_flow_enabled=True, logit_flow_hidden_dim=16,
        legacy_semantic_routes_enabled=False,
    )
    output = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 3, 3, 192, 344),
        torch.linspace(-3, 0, 4).unsqueeze(0), torch.ones(1, 4, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )

    assert output["history_image_action_logits"].shape == (1, 3, 4)
    assert output["history_image_reason_logits"].shape == (1, 3, 21)
    assert torch.equal(output["video_action_logits"], output["image_action_logits"])
    assert torch.equal(output["video_reason_logits"], output["image_reason_logits"])
    assert "logit_flow_action" in model.owner_parameters()
    assert "logit_flow_reason" in model.owner_parameters()

    with torch.no_grad():
        model.action_logit_flow.candidate_output.weight.fill_(0.1)
        model.reason_logit_flow.candidate_output.weight.fill_(0.1)
    output = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 3, 3, 192, 344),
        torch.linspace(-3, 0, 4).unsqueeze(0), torch.ones(1, 4, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    output["logit_flow_action_candidate_delta"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.action_logit_flow.parameters())
    assert all(parameter.grad is None for parameter in model.reason_logit_flow.parameters())


def test_reason_local_query_is_wired_to_history_and_exactly_protects_image_base():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        reason_local_query_enabled=True, reason_local_query_cap=0.08,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=0.0, temporal_reason_scale=0.0,
    )
    assert out["history_reason_query_tokens"].shape == (1, 14, 21, 8)
    assert out["terminal_reason_query_tokens"].shape == (1, 21, 8)
    assert out["reason_local_candidate_logits"].shape == (1, 21)
    assert torch.equal(out["video_reason_logits"], out["image_reason_logits"])
    assert "reason_local_query" in model.owner_parameters()
    out["reason_local_candidate_logits"].sum().backward()
    assert model.reason_local_query.reason_readout_weight.grad.abs().sum() > 0


def test_action_local_query_uses_terminal_visual_read_and_protects_image_base():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles, context_chunk_size=7,
        action_local_query_enabled=True, action_local_query_cap=0.08,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=0.0, temporal_reason_scale=0.0,
    )
    assert out["terminal_action_query_tokens"].shape == (1, 4, 8)
    assert not torch.equal(
        out["terminal_action_query_tokens"],
        out["terminal_query_identity"][:, :4],
    )
    assert out["action_local_candidate_logits"].shape == (1, 4)
    assert torch.equal(out["video_action_logits"], out["image_action_logits"])
    assert "action_local_query" in model.owner_parameters()
    out["action_local_candidate_logits"].sum().backward()
    assert model.action_local_query.action_readout_weight.grad.abs().sum() > 0


def test_reason_track_reader_uses_reason_specific_trajectory_groups():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(),
        dim=8,
        predicate_roles=roles,
        context_chunk_size=2,
        reason_local_query_enabled=True,
        action_local_query_enabled=True,
        track_conditioned_local_query_enabled=True,
        track_attention_topk=4,
        reason_track_attention_topk=3,
        reason_track_confidence_power=1.0,
    )
    out = model(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 3, 3, 192, 344),
        torch.linspace(-3, 0, 4).unsqueeze(0),
        torch.ones(1, 4, dtype=torch.bool),
        temporal_action_scale=0.0,
        temporal_reason_scale=0.0,
    )

    assert out["reason_trajectory_trajectory_appearance"].shape[1] == 21
    assert out["reason_track_attention"].shape[:3] == (1, 21, 4)
    assert (out["reason_track_attention"] > 0).sum(-1).max() <= 3
    assert out["reason_track_motion_rms"].shape == (1, 21)
    rerun = model.rerun_temporal_from_output(
        out,
        "time_reverse",
        temporal_action_scale=0.0,
        temporal_reason_scale=0.0,
    )
    assert rerun["reason_track_attention"].shape[:3] == (1, 21, 4)
    assert torch.isfinite(rerun["reason_local_candidate_logits"]).all()


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


def test_temporal_architecture_and_flow_caps_are_constructor_controlled():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(),
        dim=8,
        predicate_roles=roles,
        temporal_layers=1,
        temporal_heads=2,
        temporal_dropout=0.0,
        action_kappa=0.07,
        reason_kappa=0.05,
        action_flow_mix_cap=0.10,
        reason_flow_mix_cap=0.08,
    )

    assert len(model.temporal_encoder.encoder.layers) == 1
    assert model.temporal_encoder.encoder.layers[0].self_attn.num_heads == 2
    assert model.action_reader.kappa == 0.07
    assert model.reason_reader.kappa == 0.05
    assert model.action_reader.flow_mix_cap == 0.10
    assert model.reason_reader.flow_mix_cap == 0.08


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
    assert out["geometric_motion_tokens"].shape == (1, 32, 26)
    assert out["geometric_action_motion_attention"].shape == (1, 4, 32)
    assert out["geometric_reason_motion_attention"].shape == (1, 21, 32)
    torch.testing.assert_close(
        out["geometric_action_motion_attention"].sum(-1),
        torch.ones(1, 4),
    )
    assert out["geometric_action_motion_attention"].max(-1).values.min() > 0.45
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


def test_terminal_repeat_history_skips_history_dino_and_preserves_exact_fallback():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles,
        history_encoder_mode="terminal_repeat",
        geometric_flow_enabled=True,
        geometric_flow_hidden_dim=64,
    ).eval()

    def fail_if_history_dino_runs(*args, **kwargs):
        raise AssertionError("terminal_repeat must not call history DINO")

    model.context_encoder.forward = fail_if_history_dino_runs
    assert not model.owner_parameters()["history_reader"]
    output = model(
        torch.randn(2, 3, 360, 640),
        torch.randn(2, 5, 3, 192, 344),
        torch.linspace(-5, 0, 6).expand(2, -1),
        torch.ones(2, 6, dtype=torch.bool),
        temporal_action_scale=0.0,
        temporal_reason_scale=0.0,
    )

    assert output["history_encoder_mode"] == "terminal_repeat"
    assert output["history_dino_call_count"] == 0
    assert output["history_query_tokens"].shape == (2, 5, 36, 8)
    assert output["geometric_flow_field"].shape == (2, 4, 2, 45, 80)
    assert torch.equal(output["video_action_logits"], output["image_action_logits"])
    assert torch.equal(output["video_reason_logits"], output["image_reason_logits"])


def test_terminal_repeat_object_intent_uses_terminal_semantics_only():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles,
        history_encoder_mode="terminal_repeat",
        object_intent_enabled=True,
        object_tracker=_DummyObjectTracker(),
        object_intent_terminal_semantics_only=True,
    ).eval()
    captured = {}
    original = model.object_intent.forward

    def capture(*args, **kwargs):
        captured["temporal_patch_tokens"] = kwargs.get("temporal_patch_tokens")
        return original(*args, **kwargs)

    model.object_intent.forward = capture
    output = model(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 5, 3, 192, 344),
        torch.linspace(-5, 0, 6).unsqueeze(0),
        torch.ones(1, 6, dtype=torch.bool),
        temporal_action_scale=1.0,
        temporal_reason_scale=1.0,
    )
    assert captured["temporal_patch_tokens"] is None
    assert output["object_intent_track_semantics_by_frame"].shape[1] == 1
    assert output["object_intent_action_delta"].shape == (1, 4)
    assert output["object_intent_reason_delta"].shape == (1, 21)


def test_terminal_repeat_can_drive_object_intent_from_geometric_tracks():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, predicate_roles=roles,
        history_encoder_mode="terminal_repeat",
        geometric_flow_enabled=True,
        object_intent_enabled=True,
        object_tracker_mode="geometric_flow",
        object_intent_terminal_semantics_only=True,
    ).eval()
    output = model(
        torch.randn(1, 3, 360, 640),
        torch.randn(1, 5, 3, 192, 344),
        torch.linspace(-5, 0, 6).unsqueeze(0),
        torch.ones(1, 6, dtype=torch.bool),
        temporal_action_scale=1.0,
        temporal_reason_scale=1.0,
    )
    assert output["object_tracks_xy"].shape == (1, 6, 16, 2)
    assert output["object_tracks_visibility"].shape == (1, 6, 16)
    assert output["object_tracker_source"] == "geometric_flow"
    assert output["geometric_flow_field"].shape == (1, 5, 2, 45, 80)


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


def test_trajectory_can_remain_observable_without_entering_deploy_action():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7, traffic_motion_topk=4,
        traffic_trajectory_enabled=True, traffic_trajectory_deploy_enabled=False,
    )
    original = model.traffic_trajectory_head.forward

    def force_nonzero_trajectory(*args, **kwargs):
        result = original(*args, **kwargs)
        result["traffic_trajectory_delta"] = torch.full_like(
            result["traffic_trajectory_delta"], 0.03
        )
        return result

    model.traffic_trajectory_head.forward = force_nonzero_trajectory
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert torch.count_nonzero(out["traffic_trajectory_delta"]) > 0
    torch.testing.assert_close(
        out["video_action_logits"],
        out["image_action_logits"] + out["semantic_action_temporal_delta"]
        + out["geometric_action_delta"] + out["traffic_action_delta"],
    )


def test_logit_flow_deploy_delta_is_in_final_logits_and_temporal_delta_telemetry():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7, logit_flow_enabled=True,
        legacy_semantic_routes_enabled=False,
    )
    with torch.no_grad():
        model.action_logit_flow.candidate_output.weight.fill_(0.25)
        model.reason_logit_flow.candidate_output.weight.fill_(0.25)
        model.action_logit_flow.utility_output.bias.fill_(2.0)
        model.reason_logit_flow.utility_output.bias.fill_(2.0)
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 5, 3, 192, 344),
        torch.linspace(-5, 0, 6).unsqueeze(0), torch.ones(1, 6, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    torch.testing.assert_close(
        out["video_action_logits"] - out["image_action_logits"],
        out["action_temporal_delta"],
    )
    torch.testing.assert_close(
        out["video_reason_logits"] - out["image_reason_logits"],
        out["reason_temporal_delta"],
    )
    assert torch.count_nonzero(out["logit_flow_action_deploy_delta"]) > 0
    assert torch.count_nonzero(out["logit_flow_reason_deploy_delta"]) > 0


def test_target_token_flow_is_the_only_route_and_keeps_action_reason_firewall():
    roles = {
        "static_anchor": [f"p{i}" for i in range(8)],
        "dynamic_actor": [f"p{i}" for i in range(8, 24)],
        "terminal_context": [f"p{i}" for i in range(24, 32)],
    }
    model = TIDAOIAModel(
        _ImageBase(), dim=8, num_actions=4, num_reasons=21, num_predicates=32,
        predicate_roles=roles, context_chunk_size=7,
        target_token_flow_enabled=True, target_token_flow_hidden_dim=16,
        legacy_semantic_routes_enabled=False,
    )
    with torch.no_grad():
        model.target_token_action.candidate_output_weight.fill_(0.25)
        model.target_token_reason.candidate_output_weight.fill_(0.25)
        model.target_token_action.set_deployment_policy(
            torch.ones(4), torch.ones(4), torch.zeros(4), source="unit_test"
        )
        model.target_token_reason.set_deployment_policy(
            torch.ones(21), torch.ones(21), torch.zeros(21), source="unit_test"
        )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["target_token_action_predicted_terminal_token"].shape == (1, 4, 8)
    assert out["target_token_reason_predicted_terminal_token"].shape == (1, 21, 8)
    assert torch.count_nonzero(out["target_token_action_deploy_delta"]) > 0
    assert torch.count_nonzero(out["target_token_reason_deploy_delta"]) > 0
    torch.testing.assert_close(
        out["video_action_logits"] - out["image_action_logits"],
        out["target_token_action_deploy_delta"],
    )
    torch.testing.assert_close(
        out["video_reason_logits"] - out["image_reason_logits"],
        out["target_token_reason_deploy_delta"],
    )
    owners = model.owner_parameters()
    assert "target_token_action" in owners and "target_token_reason" in owners
    assert_owner_exact_cover(model, owners)
    action_from_reason = torch.autograd.grad(
        out["video_reason_logits"].sum(), owners["target_token_action"],
        allow_unused=True, retain_graph=True,
    )
    reason_from_action = torch.autograd.grad(
        out["video_action_logits"].sum(), owners["target_token_reason"],
        allow_unused=True,
    )
    assert all(value is None or torch.count_nonzero(value) == 0 for value in action_from_reason)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in reason_from_action)

    telemetry = {}
    append_supervision_tensors(
        telemetry,
        out,
        {
            "action": torch.zeros(1, 4),
            "frame_store_hit": torch.ones(1, dtype=torch.bool),
        },
    )
    assert "target_token_action_shuffled_prediction_error" in telemetry
    assert "target_token_reason_shuffled_prediction_error" in telemetry


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
        relational_event_conditioning_enabled=True,
        relational_event_conditioning_scale=0.20,
    )
    out = model(
        torch.randn(1, 3, 360, 640), torch.randn(1, 14, 3, 192, 344),
        torch.linspace(-5, 0, 15).unsqueeze(0), torch.ones(1, 15, dtype=torch.bool),
        temporal_action_scale=1.0, temporal_reason_scale=1.0,
    )
    assert out["semantic_trajectory_xy"].shape == (1, 1, 4, 15, 2)
    assert out["relational_action_attention"].shape == (1, 4, 4)
    assert out["relational_reason_attention"].shape == (1, 21, 4)
    assert out["relational_action_events"].shape == (1, 4, 12)
    assert out["relational_action_event_context"].shape == (1, 4, 8)
    assert out["relational_reason_event_context"].shape == (1, 21, 8)
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
