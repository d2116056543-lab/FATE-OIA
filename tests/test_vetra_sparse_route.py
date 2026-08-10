import torch
from vetra_test_utils import inputs, transport


def test_entmax_routes_normalize_and_have_exact_zeros():
    out = transport()(**inputs())
    for key in ("support_route", "counter_route"):
        route = out[key]
        assert torch.allclose(route.sum(-1), torch.ones_like(route[..., 0]), atol=1e-5)
        assert bool((route == 0).any())
