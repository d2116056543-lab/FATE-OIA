import torch

from fate_oia.models.tida_predicate_differential import TIDAPredicateDifferential
from fate_oia.models.tida_terminal_query_reader import TIDATerminalQueryReader


def test_region_masks_measure_attention_mass_not_region_average():
    masks = TIDATerminalQueryReader._region_masks((4, 8), torch.device("cpu"), torch.float32)
    uniform_attention = torch.full((1, 1, 32), 1.0 / 32.0)
    mass = torch.einsum("bqn,rn->bqr", uniform_attention, masks)

    assert torch.allclose(mass[..., 4], torch.ones_like(mass[..., 4]))
    assert 0.0 < mass[..., 0].item() < 1.0
    assert 0.0 < mass[..., 1].item() < 1.0
    assert torch.all((masks == 0) | (masks == 1))


def test_predicate_differential_outputs_region_derivatives():
    module = TIDAPredicateDifferential(dim=8, predicate_names=[f"p{i}" for i in range(32)], roles={"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]})
    out = module(
        torch.randn(2, 14, 32, 8), torch.randn(2, 32, 8), torch.randn(2, 32, 8),
        torch.linspace(-5, 0, 15).repeat(2, 1), torch.ones(2, 15, dtype=torch.bool),
        torch.rand(2, 14, 32, 5), torch.rand(2, 32, 5), torch.rand(2, 32),
    )
    assert out["predicate_region_mass"].shape == (2, 15, 32, 5)
    assert out["predicate_region_mass_velocity"].shape == (2, 32, 5)
    assert out["predicate_differential_state"].shape == (2, 32, 8)
