import torch

from fate_oia.models.cast_evidence_graph import CastEvidenceGraph


def test_evidence_graph_shapes_and_sparse_edges():
    model = CastEvidenceGraph(dim=32, num_labels=25, num_sets=16, topk_edges=8)
    label_nodes = torch.randn(2, 25, 32)
    label_evidence = torch.randn(2, 25, 32)
    attn = torch.softmax(torch.randn(2, 25, 32), dim=-1)
    set_nodes = torch.randn(2, 16, 32)
    text_sim = torch.eye(25)
    out = model(label_nodes, label_evidence, attn, set_nodes, text_sim)
    assert out["updated_label_nodes"].shape == (2, 25, 32)
    assert out["updated_set_nodes"].shape == (2, 16, 32)
    assert out["edge_weights"].shape == (2, 41, 41)
    assert out["reason_to_set_logits"].shape == (2, 21, 16)
    assert (out["edge_weights"] <= 1e-7).sum().item() > 0
    assert "reason_to_set_mass" in out["graph_stats"]
