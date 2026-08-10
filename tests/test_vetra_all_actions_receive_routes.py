import torch

from vetra_test_utils import inputs, transport


def test_every_action_has_non_null_support_and_counter_route_capacity():
    model = transport()
    with torch.no_grad():
        model.null_key.fill_(-20)
    output = model(**inputs(batch=4), alpha=1.0)
    assert ((1 - output["support_route"][..., -1]).sum(0) > 0).all()
    assert ((1 - output["counter_route"][..., -1]).sum(0) > 0).all()
