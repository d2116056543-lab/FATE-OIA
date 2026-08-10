import torch
from vetra_test_utils import inputs, transport


def test_predicate_identity_changes_keys_not_visual_values():
    model, data = transport(), inputs()
    out1 = model(**data)
    data["predicate_tokens"] = data["predicate_tokens"].roll(1, 1)
    out2 = model(**data)
    assert torch.equal(out1["support_factor_values"], out2["support_factor_values"])
    assert not torch.equal(out1["support_route"], out2["support_route"])
