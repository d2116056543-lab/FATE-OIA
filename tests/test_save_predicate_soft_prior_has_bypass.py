import pytest
import torch

from fate_oia.models.save_action_evidence import (
    SAVEActionEvidence,
    build_predicate_soft_prior,
)


def _staged_equivalent_at_progress_zero(
    model,
    nodes,
    global_field,
    detail_field,
    base_logits,
    *,
    calalign_action_attention,
    predicate_map,
    predicate_candidate_weight,
    predicate_reliability,
):
    global_read = model.read_global(nodes, global_field)
    evidence = model.read_detail(
        global_read,
        detail_field,
        calalign_action_attention=calalign_action_attention,
        predicate_map=predicate_map,
        predicate_candidate_weight=predicate_candidate_weight,
        predicate_reliability=predicate_reliability,
    )
    kappa = model._kappa(base_logits)
    action_evidence_bounded = kappa.view(1, -1) * torch.tanh(
        evidence["action_evidence_raw"]
        / kappa.view(1, -1).clamp_min(torch.finfo(base_logits.dtype).tiny)
    )
    gain = torch.sigmoid(model.evidence_gain_raw).to(base_logits).view(1, -1)
    auxiliary_bounded = kappa.view(1, -1) * torch.tanh(
        evidence["action_evidence_raw"]
        / kappa.view(1, -1).clamp_min(torch.finfo(base_logits.dtype).tiny)
    )
    return {
        "action_nodes_base": nodes,
        "action_logits_base": base_logits,
        "action_logits_visual": base_logits,
        **evidence,
        "action_evidence_bounded": action_evidence_bounded,
        "action_evidence_delta_unramped": action_evidence_bounded,
        "action_evidence_delta": torch.zeros_like(action_evidence_bounded),
        "action_logits_evidence_aux": base_logits.detach() + auxiliary_bounded,
        "action_logits_evidence_auxiliary": base_logits.detach() + auxiliary_bounded,
        "action_evidence_aux_raw": evidence["action_evidence_raw"],
        "action_evidence_aux_bounded": auxiliary_bounded,
        "action_logits_final": base_logits,
        "action_correction_kappa": kappa.view(1, -1),
        "action_evidence_gain": gain,
        "action_credit_ramp": base_logits.new_zeros(()),
        "action_logit_uncapped_final": base_logits,
    }


def _assert_full_tensor_contract(actual, expected):
    assert set(actual) == set(expected)
    for key in expected:
        assert isinstance(actual[key], torch.Tensor), key
        assert actual[key].shape == expected[key].shape, key
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_legacy_forward_full_contract_matches_staged_mapping_alias_and_auxiliary_paths():
    torch.manual_seed(41)
    model = SAVEActionEvidence(dim=8, action_dim=2, num_heads=2)
    nodes = torch.randn(1, 2, 8)
    global_field = torch.randn(1, 3600, 8)
    detail_field = torch.randn(1, 3600, 8)
    base_logits = torch.randn(1, 2)
    base_attention = torch.full((1, 2, 3600), 1.0 / 3600)
    predicate_map = torch.rand(1, 3, 3600)
    predicate_candidate_weight = torch.tensor(
        [[[0.20, 0.30, 0.40, 0.10], [0.10, 0.20, 0.30, 0.40]]]
    )
    predicate_reliability = torch.tensor([[0.8, 0.9, 1.0]])

    expected = _staged_equivalent_at_progress_zero(
        model,
        nodes,
        global_field,
        detail_field,
        base_logits,
        calalign_action_attention=base_attention,
        predicate_map=predicate_map,
        predicate_candidate_weight=predicate_candidate_weight,
        predicate_reliability=predicate_reliability,
    )
    canonical = model(
        nodes,
        global_field,
        detail_field,
        base_logits,
        progress=0.0,
        calalign_action_attention=base_attention,
        predicate_map=predicate_map,
        predicate_candidate_weight=predicate_candidate_weight,
        predicate_reliability=predicate_reliability,
    )
    _assert_full_tensor_contract(canonical, expected)
    torch.testing.assert_close(canonical["action_logits_final"], base_logits, atol=0, rtol=0)
    torch.testing.assert_close(
        canonical["action_evidence_delta"], torch.zeros_like(base_logits), atol=0, rtol=0
    )

    mapping_inputs = {
        "action_nodes_base": nodes,
        "global_field": global_field,
        "detail_field": detail_field,
        "action_logits_base": base_logits,
        "calalign_action_attention": base_attention,
        "predicate_map": predicate_map,
        "predicate_candidate_weight": predicate_candidate_weight,
        "predicate_reliability": predicate_reliability,
    }
    mapping_output = model(mapping_inputs, progress=0.0)
    _assert_full_tensor_contract(mapping_output, canonical)

    alias_output = model(
        action_nodes=nodes,
        global_field=global_field,
        detail_field=detail_field,
        action_logits_calalign=base_logits,
        label_attention=base_attention,
        predicate_action_map=predicate_map,
        predicate_candidate_weight=predicate_candidate_weight,
        predicate_action_reliability=predicate_reliability,
        progress=0.0,
    )
    _assert_full_tensor_contract(alias_output, canonical)


def test_entmax_candidate_distribution_preserves_final_null_and_named_mass():
    global_attention = torch.full((1, 2, 5), 0.2)
    predicate_map = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0]]]
    )
    reliability = torch.ones(1, 3)
    named_candidate_weight = torch.tensor(
        [[[0.20, 0.30, 0.50, 0.00], [0.10, 0.20, 0.70, 0.00]]]
    )
    null_candidate_weight = torch.zeros(1, 2, 4)
    null_candidate_weight[..., -1] = 1.0

    common = {
        "predicate_map": predicate_map,
        "predicate_reliability": reliability,
        "unnamed_epsilon": 1e-3,
        "unnamed_global_weight": 0.0,
        "unnamed_calalign_weight": 0.0,
    }
    named = build_predicate_soft_prior(
        global_attention,
        predicate_candidate_weight=named_candidate_weight,
        **common,
    )
    null = build_predicate_soft_prior(
        global_attention,
        predicate_candidate_weight=null_candidate_weight,
        **common,
    )

    expected_named = torch.tensor(
        [[[0.20, 0.30, 0.50, 0.00, 0.00], [0.10, 0.20, 0.70, 0.00, 0.00]]]
    )
    torch.testing.assert_close(
        named["predicate_prior_named"], expected_named, atol=0, rtol=0
    )
    torch.testing.assert_close(
        null["predicate_prior_named"],
        torch.zeros_like(null["predicate_prior_named"]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        null["predicate_prior_unnamed"],
        named["predicate_prior_unnamed"],
        atol=0,
        rtol=0,
    )
    assert torch.all(null["predicate_prior_unnamed"] > 0)


def test_predicate_candidate_weight_rejects_non_distribution_values():
    with pytest.raises(ValueError, match="entmax distribution"):
        build_predicate_soft_prior(
            torch.full((1, 2, 5), 0.2),
            predicate_map=torch.ones(1, 3, 5),
            predicate_candidate_weight=torch.zeros(1, 2, 4),
        )


def test_predicate_soft_prior_keeps_all_patches_readable_with_nonzero_unnamed_bypass():
    torch.manual_seed(23)
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4)
    patches = 3600
    predicate_map = torch.zeros(1, 21, patches)
    predicate_weights = torch.zeros(1, 4, 22)
    predicate_weights[..., -1] = 1.0
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
    predicate_weights[..., -1] = 1.0
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
