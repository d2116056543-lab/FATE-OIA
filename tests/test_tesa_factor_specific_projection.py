import torch

from fate_oia.models.meter_semantic_action import FactorSpecificActionTransport


def test_factor_projection_is_index_specific_rank16() -> None:
    module = FactorSpecificActionTransport(dim=32, rank=16)
    assert module.factor_down.shape == (21, 16, 32)
    assert module.factor_up.shape == (21, 32, 16)
    assert not torch.equal(module.factor_down[0], module.factor_down[1])
