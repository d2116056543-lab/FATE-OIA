from fate_oia.engine.train_tida_oia import training_epoch_indices


def test_evaluation_only_runs_exactly_one_epoch():
    assert list(training_epoch_indices(0, 8, evaluation_only=True)) == [0]
    assert list(training_epoch_indices(4, 8, evaluation_only=True)) == [4]


def test_training_keeps_full_requested_epoch_range():
    assert list(training_epoch_indices(2, 5, evaluation_only=False)) == [2, 3, 4]
