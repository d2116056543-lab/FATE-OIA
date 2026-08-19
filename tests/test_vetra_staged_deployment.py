import pytest

from fate_oia.engine.export_vetra_from_scratch_deploy import resolve_fit_splits


def test_action_and_reason_fit_splits_are_independent():
    action, reason = resolve_fit_splits(
        ["train_calib", "train_audit"], ["train_calib"]
    )
    assert action == ("train_calib", "train_audit")
    assert reason == ("train_calib",)


def test_reason_fit_splits_default_to_action_for_backward_compatibility():
    action, reason = resolve_fit_splits(["train_calib"], None)
    assert action == ("train_calib",)
    assert reason == action


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (["test"], ["train_calib"]),
        (["train_calib"], ["test"]),
        ([], ["train_calib"]),
        (["train_calib"], []),
    ],
)
def test_deployment_fit_splits_reject_test_and_empty_policies(action, reason):
    with pytest.raises(ValueError, match="test|non-empty"):
        resolve_fit_splits(action, reason)
