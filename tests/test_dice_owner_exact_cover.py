from fate_oia.losses.dice_loss_registry import DICELossRegistry


def test_loss_owner_names_are_exact_and_nonduplicated():
    assert set(DICELossRegistry.weights)=={"action_asl","rank_sketch","rank_protect","license","effect","delta"}
