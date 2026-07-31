import torch

from fate_oia.models.meter_meta_adapters import HECASharedPrivateAdapters


def test_zero_initialized_shared_private_adapters_are_exact_identity() -> None:
    module = HECASharedPrivateAdapters(dim=8, action_dim=4, reason_dim=21, rank=3)
    nodes = torch.randn(2, 25, 8)
    out = module(nodes)
    assert torch.equal(out["shared_nodes"], nodes)
    assert torch.equal(out["action_nodes"], nodes[:, :4])
    assert torch.equal(out["reason_nodes"], nodes[:, 4:])

