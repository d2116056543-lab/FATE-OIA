from pathlib import Path

import torch

from fate_oia.models.precise_category_decoder import PRECISECategoryDecoder
from fate_oia.utils.precise_schema import load_action_semantics, load_reason_semantics


ROOT = Path(__file__).resolve().parents[1]


def test_category_decoder_first_pass_contract_and_compositional_side_queries():
    action_schema = load_action_semantics(ROOT / "configs" / "precise_action_semantics.yaml")
    decoder = PRECISECategoryDecoder(load_reason_semantics(ROOT / "configs" / "precise_reason_semantics.yaml"), action_schema)
    action_context = torch.randn(2, 435, 384)
    reason_context = torch.randn(2, 435, 384)
    output = decoder.first_pass(decoder.action_queries(), decoder.reason_queries(), action_context, reason_context)
    assert output["action_tokens_direct"].shape == (2, 4, 384)
    assert output["reason_tokens_direct"].shape == (2, 21, 384)
    assert output["action_logits_direct"].shape == (2, 4)
    assert output["reason_logits_direct"].shape == (2, 21)
    queries = decoder.action_queries()
    assert torch.allclose(queries[2] - decoder.left_embedding, queries[3] - decoder.right_embedding)
    assert [row["name"] for row in decoder.action_schema] == ["forward", "stop", "left", "right"]
