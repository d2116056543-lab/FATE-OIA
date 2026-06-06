import torch

from fate_oia.models.cage_evidence_retriever import CAGEEvidenceRetriever
from fate_oia.models.cage_dynamic_transport import CAGEDynamicTransport
from fate_oia.models.cage_reason_reliability import CAGEReasonReliability
from fate_oia.models.cage_oia_model import CAGEOIAFeatureModel
from fate_oia.losses.cage_losses import selected_vs_random_margin_loss


def test_label_specific_evidence_retriever_outputs_per_label_evidence():
    torch.manual_seed(0)
    retriever = CAGEEvidenceRetriever(hidden_dim=32, num_labels=25, num_heads=4, topk=5)
    tokens = torch.randn(2, 64, 32)
    label_queries = torch.randn(25, 32)
    out = retriever(tokens, label_queries)

    assert out["evidence_state"].shape == (2, 25, 32)
    assert out["evidence_scores"].shape == (2, 25, 64)
    assert out["topk_indices"].shape == (2, 25, 5)
    assert torch.allclose(out["evidence_scores"].sum(dim=-1), torch.ones(2, 25), atol=1e-5)
    assert not torch.allclose(out["evidence_state"][:, 0], out["evidence_state"][:, 1])


def test_dynamic_transport_emits_typed_edges_and_updated_label_states():
    torch.manual_seed(1)
    transport = CAGEDynamicTransport(hidden_dim=32, action_dim=4, reason_dim=21, num_steps=2)
    label_state = torch.randn(2, 25, 32)
    base_logits = torch.randn(2, 25)
    prior = torch.zeros(25, 25)
    out = transport(label_state, base_logits=base_logits, cooccur_prior=prior)

    assert out["updated_label_state"].shape == (2, 25, 32)
    for key in ["A_A", "A_R", "R_A", "R_R"]:
        assert key in out["typed_edges"]
        assert out["typed_edges"][key].shape[0] == 2
    assert out["edge_matrix"].shape == (2, 25, 25)
    assert torch.isfinite(out["edge_matrix"]).all()


def test_reason_reliability_is_per_reason_and_uses_evidence_support():
    torch.manual_seed(2)
    reliability = CAGEReasonReliability(reason_dim=21, hidden_dim=16)
    reason_logits = torch.randn(2, 21)
    evidence_confidence = torch.rand(2, 21)
    selected_drop = torch.rand(2, 21)
    label_frequency = torch.linspace(0.01, 1.0, 21)
    out = reliability(reason_logits, evidence_confidence, selected_drop, label_frequency)

    assert out["reason_reliability"].shape == (2, 21)
    assert out["reason_reliability"].min() >= 0
    assert out["reason_reliability"].max() <= 1
    changed = reliability(reason_logits, evidence_confidence * 0.0, selected_drop * 0.0, label_frequency)["reason_reliability"]
    assert not torch.allclose(out["reason_reliability"], changed)


def test_selected_vs_random_margin_loss_rewards_causal_evidence():
    selected_drop = torch.tensor([[0.30, 0.10], [0.20, 0.05]])
    random_drop = torch.tensor([[0.05, 0.05], [0.05, 0.05]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    loss_good = selected_vs_random_margin_loss(selected_drop, random_drop, positive_mask=mask, margin=0.05)
    loss_bad = selected_vs_random_margin_loss(random_drop, selected_drop, positive_mask=mask, margin=0.05)

    assert loss_good < loss_bad
    assert loss_bad > 0


def test_cage_model_forward_is_branch_safe_and_label_specific():
    torch.manual_seed(3)
    model = CAGEOIAFeatureModel(input_dim=32, hidden_dim=32, action_dim=4, reason_dim=21, evidence_topk=4)
    tokens = torch.randn(2, 80, 32)
    out = model(tokens)

    required = [
        "action_logits",
        "reason_logits",
        "base_action_logits",
        "base_reason_logits",
        "transport_action_logits",
        "transport_reason_logits",
        "action_gate",
        "reason_reliability",
        "selected_vs_random_ready",
        "evidence",
        "transport",
    ]
    for key in required:
        assert key in out
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["action_gate"].min() >= 0
    assert out["action_gate"].max() <= 1
    assert out["evidence"]["evidence_state"].shape[1] == 25
