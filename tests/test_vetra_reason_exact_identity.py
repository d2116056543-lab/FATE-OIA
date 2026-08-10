import torch

from vetra_test_utils import build_model, fake_base


def test_reason_logits_are_bit_exact_for_all_ablations():
    model = build_model()
    base = fake_base(batch=2)
    variants = (
        {}, {"semantic_shuffle": True}, {"visual_shuffle": True},
        {"force_null_only": True}, {"named_factors_off": True},
        {"unnamed_factors_off": True}, {"support_route_off": True},
        {"counter_route_off": True}, {"predicate_off": True},
        {"reliability_off": True},
    )
    for variant in variants:
        out = model.decode_base_output(base, alpha=1.0, **variant)
        assert torch.equal(out["reason_logits_base"], out["reason_logits_final"])

