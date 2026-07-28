import torch

from fate_oia.models.meter_meta_adapters import METERFactorMetaAdapters


def test_meta_adapter_zero_up_is_zero_output_but_down_is_initialized() -> None:
    module = METERFactorMetaAdapters(factor_dim=21, dim=16, rank=4)
    output = module(torch.randn(2, 21, 16))
    assert torch.equal(output, torch.zeros_like(output))
    assert float(module.down.abs().sum()) > 0
