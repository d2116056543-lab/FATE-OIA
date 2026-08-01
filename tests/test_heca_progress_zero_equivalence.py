import torch

from fate_oia.models.meter_meta_adapters import HECASharedPrivateAdapters


def test_zero_initialized_shared_private_adapters_are_exact_identity() -> None:
    module = HECASharedPrivateAdapters(dim=8, action_dim=4, reason_dim=21, rank=3)
    nodes = torch.randn(2, 25, 8)
    out = module(nodes)
    assert torch.equal(out["shared_nodes"], nodes)
    assert torch.equal(out["action_nodes"], nodes[:, :4])
    assert torch.equal(out["reason_nodes"], nodes[:, 4:])


def test_adapter_only_reason_route_matches_reason_nodes_in_forward_value() -> None:
    torch.manual_seed(23)
    module = HECASharedPrivateAdapters(dim=8, action_dim=4, reason_dim=21, rank=3)
    with torch.no_grad():
        module.shared_adapter.up.weight.normal_(std=0.02)
        module.reason_private_adapter.up.weight.normal_(std=0.02)
    out = module(torch.randn(2, 25, 8))

    torch.testing.assert_close(out["reason_nodes_private"], out["reason_nodes"])
