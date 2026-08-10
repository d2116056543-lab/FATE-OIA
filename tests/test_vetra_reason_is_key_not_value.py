import torch
from vetra_test_utils import inputs, transport


def test_reason_changes_keys_but_never_factor_values():
    model, data = transport(), inputs()
    out1 = model(**data)
    data["reason_nodes_primary"] = data["reason_nodes_primary"] + 3
    out2 = model(**data)
    assert torch.equal(out1["support_factor_values"], out2["support_factor_values"])
    assert not torch.equal(out1["support_route"], out2["support_route"])
