import torch

from fate_oia.models.aie_reason_rereader import AIEReasonRereader


def test_action_prior_measurably_changes_private_attention_under_large_field_scale():
    torch.manual_seed(4)
    module = AIEReasonRereader(
        dim=8,
        reason_dim=1,
        action_dim=1,
        probes_per_action=1,
        num_predicates=1,
        num_layers=1,
    )
    with torch.no_grad():
        module.action_query.weight.zero_(); module.action_query.bias.zero_()
        module.action_key.weight.zero_(); module.action_key.bias.zero_()
        module.reason_query.weight.copy_(torch.eye(8)); module.reason_query.bias.zero_()
        module.field_keys[0].weight.copy_(torch.eye(8)); module.field_keys[0].bias.zero_()
    reason_nodes = torch.randn(1, 1, 8)
    field = torch.randn(1, 1, 20, 8) * 1000
    evidence = torch.randn(1, 1, 1, 8)
    # A realistic diffuse prior: every absolute probability is below exp(-1.5).
    # Directly clamping log(p) therefore erases all spatial variation.
    maps = torch.full((1, 1, 1, 20), 0.9 / 19)
    maps[..., 0] = 0.1
    contribution = torch.ones(1, 1, 1)
    predicate_attention = torch.full((1, 1, 20), 1 / 20)
    predicate_probs = torch.ones(1, 1)
    primary = torch.zeros(1, 1)
    enabled = module(
        reason_nodes, field, evidence, maps, contribution,
        predicate_attention, predicate_probs, primary,
        action_prior_enabled=True, predicate_prior_enabled=False,
    )
    disabled = module(
        reason_nodes, field, evidence, maps, contribution,
        predicate_attention, predicate_probs, primary,
        action_prior_enabled=False, predicate_prior_enabled=False,
    )
    effect = (enabled["reason_private_attention"] - disabled["reason_private_attention"]).abs().sum()
    assert effect > 1e-3
    assert enabled["reason_action_prior_bias_rms"] > 0
    assert enabled["reason_visual_score_rms"] > 0
