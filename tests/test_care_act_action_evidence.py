from __future__ import annotations

import torch

from fate_oia.models.care_action_evidence_experts import (
    ActionEvidenceExpertBank,
    ActionEvidenceRouter,
    ReferencePointSampler,
    tokens_to_patch_map,
)


def test_patch_sampling_and_action_evidence_are_not_global_mean_only():
    tokens = torch.randn(2, 45 * 80 + 1, 384)
    patch = tokens_to_patch_map(tokens, image_height=360, image_width=640, patch_size=8)
    sampler = ReferencePointSampler(dim=384, points_per_query=4)
    ref = torch.rand(2, 4, 4, 2)
    sampled = sampler(patch, ref)
    assert sampled.shape == (2, 4, 4, 384)

    bank = ActionEvidenceExpertBank(dim=384, action_dim=4)
    action_tokens = torch.randn(2, 4, 384)
    base_action = torch.randn(2, 4)
    out1 = bank(action_tokens, tokens, base_action_logits=base_action, structured=[None, None])
    tokens2 = tokens.clone()
    tokens2[:, 100:120] += 5.0
    out2 = bank(action_tokens, tokens2, base_action_logits=base_action, structured=[None, None])
    assert out1["action_evidence_context"].shape == (2, 4, 384)
    assert out1["action_evidence_delta_raw"].shape == (2, 4)
    assert not torch.allclose(out1["action_evidence_delta_raw"], out2["action_evidence_delta_raw"])


def test_action_evidence_router_enforces_top2_per_action():
    router = ActionEvidenceRouter(dim=384, action_dim=4, top_k=2)
    out = router(torch.randn(3, 4, 384), torch.randn(3, 4), torch.rand(3, 4))
    assert out["expert_gate"].shape == (3, 4, 5)
    assert out["expert_route_mask"].sum(-1).max().item() <= 2
    assert out["expert_route_probs"].shape == (3, 4, 5)
