import pytest
import torch
from torch import nn

from fate_oia.models.acpr_label_trunk import ACPRLabelTrunk

try:
    from fate_oia.models.save_reason_decoder import SAVEPrivateReasonDecoder
except ImportError:
    SAVEPrivateReasonDecoder = None


def _foundation_with_all_reason_primitives() -> ACPRLabelTrunk:
    foundation = ACPRLabelTrunk(dim=8, action_dim=2, reason_dim=3)
    foundation.reason_norm = nn.LayerNorm(8)
    with torch.no_grad():
        for index, parameter in enumerate(foundation.parameters(), start=1):
            parameter.fill_(index / 100.0)
    return foundation


def test_private_reason_decoder_copies_full_calalign_reason_initialization_values() -> None:
    if SAVEPrivateReasonDecoder is None:
        pytest.fail("SAVEPrivateReasonDecoder is not implemented")

    torch.manual_seed(7)
    foundation = _foundation_with_all_reason_primitives()
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    decoder.initialize_from_foundation(foundation)

    assert decoder.reason_queries.shape == (3, 8)
    torch.testing.assert_close(decoder.reason_queries, foundation.label_queries[2:])
    torch.testing.assert_close(decoder.query_projection.weight, foundation.query_proj.weight)
    torch.testing.assert_close(decoder.query_projection.bias, foundation.query_proj.bias)
    torch.testing.assert_close(decoder.key_projection.weight, foundation.key_proj.weight)
    torch.testing.assert_close(decoder.key_projection.bias, foundation.key_proj.bias)
    torch.testing.assert_close(decoder.value_projection.weight, foundation.value_proj.weight)
    torch.testing.assert_close(decoder.value_projection.bias, foundation.value_proj.bias)
    source_q, source_k, source_v = foundation.label_self_attn.in_proj_weight.chunk(3)
    source_q_bias, source_k_bias, source_v_bias = foundation.label_self_attn.in_proj_bias.chunk(3)
    torch.testing.assert_close(decoder.reason_self_attention.q_proj.weight, source_q)
    torch.testing.assert_close(decoder.reason_self_attention.k_proj.weight, source_k)
    torch.testing.assert_close(decoder.reason_self_attention.v_proj.weight, source_v)
    torch.testing.assert_close(decoder.reason_self_attention.q_proj.bias, source_q_bias)
    torch.testing.assert_close(decoder.reason_self_attention.k_proj.bias, source_k_bias)
    torch.testing.assert_close(decoder.reason_self_attention.v_proj.bias, source_v_bias)
    torch.testing.assert_close(
        decoder.reason_self_attention.out_proj.weight,
        foundation.label_self_attn.out_proj.weight,
    )
    torch.testing.assert_close(
        decoder.reason_self_attention.out_proj.bias,
        foundation.label_self_attn.out_proj.bias,
    )
    torch.testing.assert_close(decoder.reason_norm.weight, foundation.reason_norm.weight)
    torch.testing.assert_close(decoder.reason_norm.bias, foundation.reason_norm.bias)
    torch.testing.assert_close(decoder.classifier.weight, foundation.logit_head.weight)
    torch.testing.assert_close(decoder.classifier.bias, foundation.logit_head.bias)
    assert decoder.reason_queries.data_ptr() != foundation.label_queries.data_ptr()
    assert decoder.reason_self_attention is not foundation.label_self_attn

    output = decoder(
        reason_logits_clean=torch.randn(1, 3),
        global_field=torch.randn(1, 3600, 8),
        detail_field=torch.randn(1, 3600, 8),
        factor_measurement_token=torch.randn(1, 3, 8),
        factor_evidence_map=torch.rand(1, 3, 3600),
        factor_reliability=torch.ones(1, 3),
        progress=1.0,
    )
    assert output["reason_logits_benchmark"].shape == (1, 3)
    assert output["reason_logits_private_direct"].shape == (1, 3)
    assert output["reason_embedding_private"].shape == (1, 3, 8)


@pytest.mark.parametrize(
    "missing",
    [
        "label_queries",
        "query_proj",
        "key_proj",
        "value_proj",
        "label_self_attn",
        "reason_norm",
        "logit_head",
    ],
)
def test_private_reason_initialization_fails_closed_when_a_plan_primitive_is_missing(
    missing: str,
) -> None:
    foundation = _foundation_with_all_reason_primitives()
    delattr(foundation, missing)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)

    with pytest.raises(ValueError, match=missing):
        decoder.initialize_from_foundation(foundation)
