from fate_oia.engine.train_acpr_oia import get_pair_weights


def test_hardpair_schedule_epochs():
    assert get_pair_weights(0, 1.0) == (0.0, 0.0)
    assert get_pair_weights(3, 0.0) == (0.05, 0.01)
    assert get_pair_weights(6, 0.0) == (0.05, 0.01)
    assert get_pair_weights(11, 0.10) == (0.05, 0.01)
    assert get_pair_weights(11, 0.01) == (0.05, 0.01)

