from __future__ import annotations


def cooldown_allowed_from_train_calib(epoch: int, min_epoch: int = 8) -> bool:
    return int(epoch) >= int(min_epoch)


def assert_test_metrics_not_used_for_training(state_source: str) -> None:
    if str(state_source).lower() == "test":
        raise ValueError("ACPR-GEM forbids using test metrics for teacher/LR/cooldown/stopping")
