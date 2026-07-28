import torch

from fate_oia.models.meter_signed_factors import METERsignedFactors


def test_signed_factor_maps_normalize_over_patches_and_null() -> None:
    torch.manual_seed(3)
    module = METERsignedFactors(dim=32, factor_dim=21, num_layers=3, rank=8)
    factors = torch.randn(2, 21, 32)
    patches = torch.randn(2, 3, 20, 32)
    output = module(factors, patches, progress=1.0)

    assert output["factor_support_maps_by_layer"].shape == (2, 21, 3, 20)
    assert output["factor_counter_maps_by_layer"].shape == (2, 21, 3, 20)
    assert output["factor_support_map"].shape == (2, 21, 20)
    assert output["factor_counter_map"].shape == (2, 21, 20)
    assert output["factor_core_tokens"].shape == (2, 21, 32)
    assert output["factor_action_tokens"].shape == (2, 21, 32)
    assert torch.allclose(
        output["factor_support_maps_by_layer"].sum(-1) + output["factor_support_null_by_layer"],
        torch.ones(2, 21, 3),
        atol=1e-5,
    )
    assert torch.allclose(
        output["factor_counter_maps_by_layer"].sum(-1) + output["factor_counter_null_by_layer"],
        torch.ones(2, 21, 3),
        atol=1e-5,
    )
    assert torch.isfinite(output["factor_reliability"]).all()
    assert (output["factor_layer_weights"] <= 0.85).all()


def test_factor_meta_reason_bridge_is_omega_controlled() -> None:
    torch.manual_seed(4)
    module = METERsignedFactors(dim=16, factor_dim=21, num_layers=3, rank=4)
    factors = torch.randn(1, 21, 16, requires_grad=True)
    patches = torch.randn(1, 3, 12, 16)
    output = module(factors, patches, progress=1.0, meta_share_weight=torch.zeros(21))
    output["factor_to_reason_tokens"].sum().backward()
    no_share = module.meta_adapters.parameter_grad_norm()

    module.zero_grad(set_to_none=True)
    output = module(factors.detach(), patches, progress=1.0, meta_share_weight=torch.ones(21))
    output["factor_to_reason_tokens"].sum().backward()
    share = module.meta_adapters.parameter_grad_norm()
    assert no_share == 0.0
    assert share > 0.0


def test_signed_factor_null_remains_nontrivial_on_full_patch_grid() -> None:
    torch.manual_seed(17)
    module = METERsignedFactors(dim=32, factor_dim=21, num_layers=3, rank=8)
    factors = torch.randn(1, 21, 32)
    patches = torch.randn(1, 3, 3600, 32)
    output = module(factors, patches, progress=1.0)

    for prefix in ("support", "counter"):
        null = output[f"factor_{prefix}_null_by_layer"]
        patch_map = output[f"factor_{prefix}_maps_by_layer"]
        assert 0.01 < float(null.mean()) < 0.80
        torch.testing.assert_close(
            patch_map.sum(-1) + null,
            torch.ones_like(null),
            atol=1e-5,
            rtol=0,
        )
    cosine = torch.nn.functional.cosine_similarity(
        output["factor_support_map"],
        output["factor_counter_map"],
        dim=-1,
    ).mean()
    assert float(cosine) < 0.98
