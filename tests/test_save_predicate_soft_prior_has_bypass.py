import torch

from fate_oia.models.save_action_evidence import (
    SAVEActionEvidence,
    build_predicate_soft_prior,
)


def test_null_candidate_competes_with_named_candidates_in_shared_normalization():
    global_attention = torch.full((1, 2, 5), 0.2)
    predicate_map = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0]]]
    )
    reliability = torch.ones(1, 3)
    weak_null_logits = torch.zeros(1, 2, 4)
    weak_null_logits[..., -1] = -20.0
    strong_null_logits = torch.zeros(1, 2, 4)
    strong_null_logits[..., -1] = 20.0

    common = {
        "predicate_map": predicate_map,
        "predicate_reliability": reliability,
        "unnamed_epsilon": 1e-3,
        "unnamed_global_weight": 0.0,
        "unnamed_calalign_weight": 0.0,
    }
    weak_null = build_predicate_soft_prior(
        global_attention,
        predicate_candidate_weight=weak_null_logits,
        **common,
    )
    strong_null = build_predicate_soft_prior(
        global_attention,
        predicate_candidate_weight=strong_null_logits,
        **common,
    )

    assert strong_null["predicate_prior_named"].sum() > 0
    assert (
        strong_null["predicate_prior_named"].sum()
        < weak_null["predicate_prior_named"].sum() * 1e-6
    )
    torch.testing.assert_close(
        strong_null["predicate_prior_unnamed"],
        weak_null["predicate_prior_unnamed"],
        atol=0,
        rtol=0,
    )
    assert torch.all(strong_null["predicate_prior_unnamed"] > 0)


def test_predicate_soft_prior_keeps_all_patches_readable_with_nonzero_unnamed_bypass():
    torch.manual_seed(23)
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    patches = 3600
    predicate_map = torch.zeros(1, 21, patches)
    predicate_weights = torch.zeros(1, 4, 21)
    predicate_reliability = torch.zeros(1, 21)

    output = model(
        torch.randn(1, 4, 16),
        torch.randn(1, patches, 16),
        torch.randn(1, patches, 16),
        torch.randn(1, 4),
        progress=1.0,
        predicate_map=predicate_map,
        predicate_candidate_weight=predicate_weights,
        predicate_reliability=predicate_reliability,
    )

    assert torch.all(output["predicate_prior_unnamed"] > 0)
    assert torch.all(output["predicate_prior"] > 0)
    assert torch.isfinite(output["detail_attention_bias_predicate"]).all()
    assert torch.isfinite(output["action_detail_attention"]).all()
    assert torch.all(output["action_detail_attention"] > 0)


def test_epsilon_only_null_bypass_keeps_every_patch_functionally_readable():
    torch.manual_seed(37)
    model = SAVEActionEvidence(
        dim=8,
        action_dim=2,
        num_heads=2,
        unnamed_epsilon=1e-3,
        unnamed_global_weight=0.0,
        unnamed_calalign_weight=0.0,
    )
    predicate_map = torch.zeros(1, 3, 3600)
    predicate_weights = torch.zeros(1, 2, 4)
    predicate_reliability = torch.zeros(1, 3)
    output = model(
        torch.randn(1, 2, 8),
        torch.randn(1, 3600, 8),
        torch.randn(1, 3600, 8),
        torch.randn(1, 2),
        predicate_map=predicate_map,
        predicate_candidate_weight=predicate_weights,
        predicate_reliability=predicate_reliability,
    )

    torch.testing.assert_close(
        output["predicate_prior_unnamed"],
        torch.full((1, 2, 3600), 1e-3),
        atol=0,
        rtol=0,
    )
    assert torch.all(output["action_detail_attention"] > 0)
    torch.testing.assert_close(
        output["action_detail_attention"].sum(-1),
        torch.ones(1, 2),
        atol=1e-6,
        rtol=0,
    )
