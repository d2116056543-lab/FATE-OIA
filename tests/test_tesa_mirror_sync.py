import torch

from fate_oia.datasets.meter_dataset import mirror_typed_target


def test_tesa_mirror_swaps_factor_identity_and_anchor_geometry():
    target = {
        "factor_anchor_map": torch.arange(21 * 2 * 3).reshape(21, 2, 3).float(),
        "factor_anchor_valid": torch.arange(21),
        "factor_state_target": torch.arange(21),
        "factor_state_valid": torch.arange(21),
        "factor_present_valid": torch.arange(21).remainder(2).bool(),
        "factor_absent_valid": torch.arange(21).remainder(3).eq(0),
        "factor_source_complete": torch.arange(21).remainder(4).eq(0),
        "factor_observability": torch.arange(21).float(),
        "factor_observability_valid": torch.arange(21),
        "factor_source_weight": torch.arange(21).float(),
    }
    mirrored = mirror_typed_target(target)
    torch.testing.assert_close(
        mirrored["factor_anchor_map"][9],
        torch.flip(target["factor_anchor_map"][15], dims=[-1]),
    )
    assert mirrored["factor_state_target"][14].item() == 20
    assert mirrored["factor_state_target"][20].item() == 14
    for key in (
        "factor_present_valid",
        "factor_absent_valid",
        "factor_source_complete",
    ):
        torch.testing.assert_close(mirrored[key][9], target[key][15])
        torch.testing.assert_close(mirrored[key][15], target[key][9])
