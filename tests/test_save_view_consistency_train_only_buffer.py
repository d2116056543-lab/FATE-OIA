import torch

from fate_oia.losses.save_pu_losses import (
    TrainOnlyViewConsistencyBuffer,
    view_consistency_loss,
)


def test_view_consistency_buffer_updates_from_train_and_is_read_only_for_test() -> None:
    buffer = TrainOnlyViewConsistencyBuffer(num_labels=2, momentum=0.5)
    logits = torch.zeros(3, 2, requires_grad=True)
    view_logits = torch.zeros(3, 2, requires_grad=True)
    measurement = torch.zeros(3, 2, requires_grad=True)
    view_measurement = torch.ones(3, 2, requires_grad=True)

    before = {key: value.clone() for key, value in buffer.state_dict().items()}
    train_value = buffer.update(
        logits,
        view_logits,
        measurement,
        view_measurement,
        split_name="train_main",
    )
    after_train = {key: value.clone() for key, value in buffer.state_dict().items()}
    test_value = buffer.read(split_name="test")
    after_test = {key: value.clone() for key, value in buffer.state_dict().items()}

    assert not train_value.requires_grad
    assert not test_value.requires_grad
    assert not torch.equal(before["consistency_ema"], after_train["consistency_ema"])
    assert torch.equal(after_train["consistency_ema"], after_test["consistency_ema"])
    assert all(torch.equal(after_train[key], after_test[key]) for key in after_train)


def test_view_consistency_test_update_is_rejected_without_state_change() -> None:
    buffer = TrainOnlyViewConsistencyBuffer(num_labels=2)
    before = {key: value.clone() for key, value in buffer.state_dict().items()}

    try:
        buffer.update(
            torch.zeros(1, 2),
            torch.ones(1, 2),
            torch.zeros(1, 2),
            torch.ones(1, 2),
            split_name="test",
        )
    except ValueError as error:
        assert "train" in str(error)
    else:
        raise AssertionError("test data must not update the train-only buffer")

    after = buffer.state_dict()
    assert all(torch.equal(before[key], after[key]) for key in before)


def test_view_consistency_loss_is_zero_for_matching_views() -> None:
    same = view_consistency_loss(
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
    )
    changed = view_consistency_loss(
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.zeros(2, 3),
        torch.ones(2, 3),
    )

    assert float(same) == 0.0
    assert float(changed) > 0.0


def test_view_buffer_exact_subsequent_ema_counter_and_state_restore() -> None:
    buffer = TrainOnlyViewConsistencyBuffer(num_labels=2, momentum=0.5)
    zeros = torch.zeros(1, 2)
    first_view = torch.tensor([[1.0, 2.0]])
    second_view = torch.tensor([[0.0, 1.0]])

    buffer.update(zeros, first_view, zeros, zeros, split_name="train_main")
    first_expected = torch.exp(-torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(buffer.consistency_ema, first_expected)
    assert int(buffer.consistency_updates) == 1

    buffer.update(zeros, second_view, zeros, zeros, split_name="train_main")
    second_observation = torch.exp(-torch.tensor([0.0, 1.0]))
    expected = 0.5 * first_expected + 0.5 * second_observation
    torch.testing.assert_close(buffer.consistency_ema, expected)
    assert int(buffer.consistency_updates) == 2

    restored = TrainOnlyViewConsistencyBuffer(num_labels=2, momentum=0.5)
    restored.load_state_dict(buffer.state_dict())
    torch.testing.assert_close(restored.read(split_name="test"), expected)
    assert int(restored.consistency_updates) == 2
