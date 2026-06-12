import torch

from fate_oia.models.cast_oia_model import CastOIAModel


def test_cast_model_forward_with_synthetic_tokens():
    model = CastOIAModel(dim=32, use_dino=False, grid_hw=(4, 4))
    images = torch.randn(2, 3, 32, 32)
    out = model(images)
    expected = {
        "action_logits", "reason_logits", "action_set_logits", "action_set_probs",
        "action_marginal_probs", "atomic_action_logits", "pair_logits",
        "cardinality_logits", "label_attention", "label_evidence",
        "label_layer_weights", "graph_edge_weights", "reason_to_set_logits",
        "reason_reliability", "evidence_stats", "graph_stats", "action_set_stats",
    }
    assert expected.issubset(out.keys())
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["action_set_logits"].shape == (2, 16)
    assert out["label_attention"].shape[-1] == 16
