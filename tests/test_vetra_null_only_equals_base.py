import torch
from vetra_test_utils import inputs, transport


def test_null_only_produces_exact_zero_delta():
    out = transport()(**inputs(), force_null_only=True)
    assert torch.equal(out["vetra_action_delta"], torch.zeros_like(out["vetra_action_delta"]))
