from pathlib import Path

from fate_oia.engine.export_tida_image_oracle import ORACLE_TENSOR_KEYS


def test_independent_image_oracle_covers_full_frozen_branch():
    required = {
        "cls_tokens_by_layer", "patch_tokens_by_layer", "action_logits_primary", "reason_logits_primary",
        "predicate_logits", "predicate_probs", "predicate_tokens", "predicate_attention",
        "action_nodes_primary", "reason_nodes_primary", "evidence_token", "evidence_map",
        "sampling_offsets", "sampling_weights", "bounded_contribution", "action_logits_final",
        "reason_private_attention", "reason_delta", "reason_logits_final",
    }
    assert required <= set(ORACLE_TENSOR_KEYS)
    source = Path("fate_oia/engine/export_tida_image_oracle.py").read_text(encoding="utf-8")
    assert 'sys.path = [source' in source
    assert 'source_head' in source and 'source_tree' in source
