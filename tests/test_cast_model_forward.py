import torch

from fate_oia.models.cast_oia_model import CastOIAModel


def test_cast_model_forward_with_synthetic_tokens():
    model = CastOIAModel(dim=32, use_dino=False, grid_hw=(4, 4))
    images = torch.randn(2, 3, 32, 32)
    out = model(images)
    expected = {
        "action_logits", "reason_logits", "action_set_logits", "action_set_probs",
        "action_marginal_probs", "atomic_action_logits", "pair_logits",
        "main_action_logits", "main_label_logits", "main_label_attention",
        "base_action_logits", "cast_action_logits", "action_fusion_gate",
        "cardinality_logits", "label_attention", "label_evidence",
        "label_layer_weights", "graph_edge_weights", "reason_to_set_logits",
        "reason_reliability", "evidence_stats", "graph_stats", "action_set_stats",
    }
    assert expected.issubset(out.keys())
    assert out["action_logits"].shape == (2, 4)
    assert out["main_action_logits"].shape == (2, 4)
    assert out["main_label_logits"].shape == (2, 25)
    assert out["base_action_logits"].shape == (2, 4)
    assert out["cast_action_logits"].shape == (2, 4)
    assert out["action_fusion_gate"].shape == (2, 4)
    assert torch.all(out["action_fusion_gate"] >= 0.0)
    assert torch.all(out["action_fusion_gate"] <= 1.0)
    assert out["reason_logits"].shape == (2, 21)
    assert out["action_set_logits"].shape == (2, 16)
    assert out["main_label_attention"].shape[-1] == 17
    assert out["label_attention"].shape[-1] == 16


def test_cast_final_action_is_guarded_main_action_fusion():
    model = CastOIAModel(dim=32, use_dino=False, grid_hw=(4, 4))
    images = torch.randn(2, 3, 32, 32)
    out = model(images)
    expected = out["main_action_logits"] + out["action_fusion_gate"] * out["bounded_action_delta"]
    assert torch.allclose(out["action_logits"], expected, atol=1e-6)
    assert torch.allclose(out["base_action_logits"], out["main_action_logits"], atol=1e-6)
    assert torch.max(torch.abs(out["bounded_action_delta"])) <= 2.0 + 1e-6
    assert hasattr(model, "main_label_head")
    assert not hasattr(model, "base_action_head")



def test_reason_head_bias_is_not_strong_negative():
    from fate_oia.models.cast_reason_reliability import CastReasonReliability

    head = CastReasonReliability(dim=32, reason_dim=21)
    assert torch.all(head.reason_head.bias >= -1.0)
    assert torch.all(head.reason_head.bias <= 0.0)
