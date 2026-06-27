from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel


def test_model_forward_shapes_and_ledger_identity():
    model = ACPRInteractFlowPPModel(
        pretrained_weights="missing-is-ok-with-mock",
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path="configs/acpr_interactflow_state_grammar.yaml",
        action_dim=3,
        use_mock_dino=True,
    )
    frames = torch.randn(2, 15, 3, 32, 64)
    out = model(frames)
    assert out.action_logits.shape == (2, 3)
    assert out.exp29_logits.shape == (2, 29)
    assert out.predicates.predicate_logits.shape == (2, 48)
    assert out.predicates.predicate_logits_trajectory.shape == (2, 15, 48)
    assert out.predicates.predicate_evidence_maps.shape == (2, 15, 48, 45, 80)
    assert out.predicates.predicate_centroids.shape == (2, 15, 48, 2)
    assert out.predicates.predicate_corridor_mass.shape == (2, 15, 48, 4)
    assert out.predicates.transfer_gate.shape == (48,)
    assert out.flow.flow_edges.shape[-1] == 3
    assert out.ledger.global_logits.shape == (2, 3)
    assert out.ledger.gated_state_contributions.shape[-1] == 3
    assert out.ledger.benefit_gate.shape == out.ledger.gate.shape
    assert float(out.ledger.identity_error) < 1e-5
    assert "state_group_logits" in out.aux
