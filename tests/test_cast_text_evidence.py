from pathlib import Path

import torch
import yaml

from fate_oia.models.cast_ego_encoding import EgoPatchCoordinateEncoder
from fate_oia.models.cast_label_evidence import CastLabelEvidence
from fate_oia.models.cast_text_encoder import build_label_texts, HashingTextPrototypeEncoder, CastLabelQueryBuilder


def test_ontology_text_prototypes_and_queries():
    ontology = yaml.safe_load(Path("configs/cast_oia_label_ontology.yaml").read_text(encoding="utf-8"))
    texts = build_label_texts(ontology)
    assert len(texts) == 25
    assert all(not t.startswith("reason_") for t in texts)
    enc = HashingTextPrototypeEncoder(dim=32)
    proto = enc(texts)
    assert proto.shape == (25, 32)
    query = CastLabelQueryBuilder(dim=32, action_dim=4, reason_dim=21)
    out = query(texts)
    assert out["label_queries"].shape == (25, 32)
    assert out["text_similarity_matrix"].shape == (25, 25)
    out["label_queries"].sum().backward()
    assert query.text_projection.weight.grad is not None


def test_ego_encoder_and_label_specific_sparse_evidence():
    b, s, n, d = 2, 3, 16, 32
    tokens = torch.randn(b, s, n, d)
    ego = EgoPatchCoordinateEncoder(dim=d, grid_hw=(4, 4))
    changed, ego_features = ego(tokens)
    assert changed.shape == tokens.shape
    assert ego_features.shape == (n, 8)
    assert not torch.allclose(changed, tokens)
    label_queries = torch.randn(25, d, requires_grad=True)
    module = CastLabelEvidence(dim=d, num_labels=25, selected_layers=s, num_heads=4)
    out = module(label_queries, changed, ego_features)
    assert out["label_attention"].shape == (b, 25, n)
    assert out["label_evidence"].shape == (b, 25, d)
    assert out["label_layer_weights"].shape == (25, s)
    assert torch.allclose(out["label_attention"].sum(-1), torch.ones(b, 25), atol=1e-4)
    assert (out["label_attention"] <= 1e-7).sum().item() > 0
    assert "right_corridor_mass" in out["attention_stats"]
    out["label_evidence"].sum().backward()
    assert module.layer_router.grad is not None
