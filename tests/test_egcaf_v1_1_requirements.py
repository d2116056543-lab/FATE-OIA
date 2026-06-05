from __future__ import annotations

import torch

from fate_oia.models.egcaf_factor_actor import FactorActor
from fate_oia.models.egcaf_factor_bank import DrivingFactorCandidateBank
from fate_oia.models.egcaf_oia_model import EGCafOIAModel
from fate_oia.models.egcaf_sparse_topk import entmax15, sparsemax


def test_factor_bank_preserves_action_specific_anchor_factors():
    bank = DrivingFactorCandidateBank(hidden_dim=16, object_factors=2, num_actions=4)
    pyramid = []
    for action_id in range(4):
        pyramid.append({
            "P1": torch.randn(2, 16, 8, 12) + action_id,
            "P2": torch.randn(2, 16, 4, 6) + action_id,
            "P3": torch.randn(2, 16, 2, 3) + action_id,
        })
    out = bank(pyramid)
    factors = out["factors"]
    assert out["factor_bank_stats"]["action_specific_maps_preserved"] is True
    assert factors.action_ids is not None
    assert set(factors.action_ids[factors.action_ids >= 0].unique().tolist()) == {0, 1, 2, 3}


def test_entmax15_is_not_sparsemax_alias():
    scores = torch.tensor([[1.0, 0.2, -0.5, 2.0]])
    assert not torch.allclose(entmax15(scores), sparsemax(scores))
    assert torch.allclose(entmax15(scores).sum(-1), torch.ones(1), atol=1e-5)


def test_factor_actor_required_modes():
    actor = FactorActor(hidden_dim=8, action_dim=4, residual_enabled_default=False)
    embeddings = torch.randn(2, 12, 8)
    weights = torch.softmax(torch.randn(2, 4, 12), -1)
    selected = torch.topk(weights, 3, dim=-1).indices
    random_idx = torch.randint(0, 12, selected.shape)
    outs = {}
    for mode in ["all", "selected", "without-selected", "without-random"]:
        outs[mode] = actor.mode_logits(embeddings, weights, selected, mode=mode, random_indices=random_idx)
        assert outs[mode].shape == (2, 4)
    assert not torch.allclose(outs["selected"], outs["without-selected"])


def test_model_outputs_v1_1_reason_factor_attention_and_residual_off():
    model = EGCafOIAModel(hidden_dim=32, lightweight_backbone=True, residual_enabled=False)
    images = torch.randn(2, 3, 64, 96)
    out = model(images, bdd100k_scene_state=torch.zeros(2, 6), return_artifacts=True)
    assert out["reason_factor_attention"].shape[:3] == (2, 21, 4)
    assert torch.equal(out["action_final_logits"], out["action_core_logits"])
    assert "z_all" in out and "z_without_selected" in out and "z_without_random" in out
    assert out["factor_bank_stats"]["action_specific_maps_preserved"] is True

