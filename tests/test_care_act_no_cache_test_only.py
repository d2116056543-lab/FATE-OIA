from __future__ import annotations

import argparse

from fate_oia.engine.train_care_act_oia import is_full_goal_run, validate_args


def test_full_goal_and_test_only_validation():
    args = argparse.Namespace(
        epochs=24,
        max_train_samples=0,
        max_test_samples=0,
        test_only_evaluation=True,
        best_selection_split="test",
        feature_cache_enabled=False,
        token_compression="none",
    )
    validate_args(args)
    assert is_full_goal_run(args)
    args.best_selection_split = "val"
    assert not is_full_goal_run(args)
