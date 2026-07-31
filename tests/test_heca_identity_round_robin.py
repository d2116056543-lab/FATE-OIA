from fate_oia.optim.heca_optimization import identity_corruption_mode


def test_identity_corruption_rotates_once_per_optimizer_update() -> None:
    assert [identity_corruption_mode(i) for i in range(6)] == [
        "schema", "cross_sample", "state", "schema", "cross_sample", "state"
    ]

