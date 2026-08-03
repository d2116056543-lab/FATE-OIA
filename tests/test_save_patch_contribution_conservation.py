import math

import torch

from fate_oia.models.save_action_evidence import (
    SAVEActionEvidence,
    build_predicate_soft_prior,
)


def _run(dtype: torch.dtype):
    torch.manual_seed(19)
    model = SAVEActionEvidence(dim=16, action_dim=4, num_heads=4).to(dtype=dtype)
    patches = 3600
    action_nodes = torch.randn(2, 4, 16, dtype=dtype)
    global_field = torch.randn(2, patches, 16, dtype=dtype)
    detail_field = torch.randn(2, patches, 16, dtype=dtype)
    base_logits = torch.randn(2, 4, dtype=dtype)
    output = model(
        action_nodes,
        global_field,
        detail_field,
        base_logits,
        progress=1.0,
    )
    return model, detail_field, output


def _independent_reconstruction(
    model: SAVEActionEvidence,
    detail_field: torch.Tensor,
    output: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prior = build_predicate_soft_prior(output["action_global_attention"])
    query = model.detail_query(output["action_global_token"])
    key = model.detail_key(detail_field)
    scores = (
        torch.einsum("bad,bnd->ban", query, key) / math.sqrt(model.dim)
        + prior["detail_attention_bias_base"].to(detail_field)
        + prior["detail_attention_bias_predicate"].to(detail_field)
    )
    attention = torch.softmax(scores, dim=-1)
    action_value = model.patch_action_value(output["action_detail_token"])
    patch_value = model.patch_value(detail_field)
    signed_value = torch.einsum("bad,bnd->ban", action_value, patch_value)
    signed_value = signed_value / math.sqrt(model.dim)
    contribution = attention * signed_value
    return signed_value, contribution, contribution.sum(dim=-1)


def test_signed_patch_contributions_sum_to_raw_evidence_in_fp32_and_bf16():
    for dtype, tolerance in ((torch.float32, 1e-6), (torch.bfloat16, 5e-4)):
        model, detail_field, output = _run(dtype)
        expected_value, expected_contribution, expected_raw = _independent_reconstruction(
            model, detail_field, output
        )
        torch.testing.assert_close(
            output["action_patch_value"], expected_value, atol=tolerance, rtol=0
        )
        torch.testing.assert_close(
            output["action_patch_contribution"],
            expected_contribution,
            atol=tolerance,
            rtol=0,
        )
        torch.testing.assert_close(
            output["action_evidence_raw"], expected_raw, atol=tolerance, rtol=0
        )
        assert (expected_value < 0).any()
        assert (expected_value > 0).any()
