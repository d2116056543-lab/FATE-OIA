from __future__ import annotations

import torch

from fate_oia.models.score_v2_oia_model import ScoreV2OIAConfig, ScoreV2OIAModel
from fate_oia.models.semantic_label_queries import SemanticLabelQueries
from fate_oia.models.strong_label_decoder import StrongLabelDecoder


def test_semantic_label_queries_return_batch_queries() -> None:
    queries = SemanticLabelQueries(num_labels=25, dim=32, dropout=0.0)
    out = queries(batch_size=4, device=torch.device("cpu"))
    assert out.shape == (4, 25, 32)
    assert out.requires_grad


def test_strong_label_decoder_outputs_action_reason_logits_and_attention() -> None:
    decoder = StrongLabelDecoder(dim=32, action_dim=4, reason_dim=21, num_heads=4, self_layers=1)
    label_queries = torch.randn(2, 25, 32)
    tokens = torch.randn(2, 65, 32)
    out = decoder(label_queries, tokens)
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["attention"].shape[:3] == (2, 25, 65)


def test_score_v2_model_forward_uses_multi_layer_tokens() -> None:
    cfg = ScoreV2OIAConfig(dim=32, action_dim=4, reason_dim=21, n_last_blocks=3, num_heads=4, decoder_self_layers=1)
    model = ScoreV2OIAModel(cfg)
    layers = [torch.randn(2, 65, 32) for _ in range(3)]
    out = model(layers)
    assert out["logits"].shape == (2, 25)
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["layer_weights"].shape == (3,)
    assert out["score_v2_stage"] == "score_v2_patch_only"
