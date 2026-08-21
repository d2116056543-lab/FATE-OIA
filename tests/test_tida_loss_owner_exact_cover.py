import torch
from torch import nn

from fate_oia.losses.tida_loss_registry import assert_owner_exact_cover


def test_owner_parameter_sets_are_disjoint_and_complete():
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    owners = {
        "first": list(model[0].parameters()),
        "second": list(model[1].parameters()),
    }
    assert_owner_exact_cover(model, owners)
    owners["second"].append(next(model[0].parameters()))
    try:
        assert_owner_exact_cover(model, owners)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate owner was accepted")
