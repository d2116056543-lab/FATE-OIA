import torch
from pathlib import Path

from fate_oia.utils.tida_action_rank_memory import TIDAActionRankMemory


def test_action_rank_memory_is_fixed_capacity_ring() -> None:
    memory = TIDAActionRankMemory(capacity=5, num_actions=4, device="cpu")
    for start in (0, 2, 4):
        logits = torch.arange(start, start + 2, dtype=torch.float32)[:, None].repeat(1, 4)
        targets = (logits > 2).float()
        memory.enqueue(logits, targets)

    snapshot = memory.snapshot()
    assert len(memory) == 5
    assert snapshot is not None
    assert snapshot["action_logits"].shape == (5, 4)
    assert snapshot["action_target"].shape == (5, 4)
    assert set(snapshot["action_logits"][:, 0].tolist()) == {1.0, 2.0, 3.0, 4.0, 5.0}


def test_action_rank_memory_detaches_and_stays_on_requested_device() -> None:
    memory = TIDAActionRankMemory(capacity=8, num_actions=4, device="cpu")
    logits = torch.randn(3, 4, requires_grad=True)
    targets = torch.randint(0, 2, (3, 4)).float()
    memory.enqueue(logits, targets)
    snapshot = memory.snapshot()

    assert snapshot is not None
    assert snapshot["action_logits"].device.type == "cpu"
    assert not snapshot["action_logits"].requires_grad
    assert not snapshot["action_target"].requires_grad


def test_action_rank_memory_rejects_wrong_shape_and_can_reset() -> None:
    memory = TIDAActionRankMemory(capacity=8, num_actions=4, device="cpu")
    try:
        memory.enqueue(torch.randn(2, 3), torch.randn(2, 3))
    except ValueError as error:
        assert "[B,4]" in str(error)
    else:
        raise AssertionError("wrong action dimension must fail")

    memory.enqueue(torch.randn(2, 4), torch.zeros(2, 4))
    memory.reset()
    assert len(memory) == 0
    assert memory.snapshot() is None


def test_zero_capacity_explicitly_disables_cross_update_memory() -> None:
    memory = TIDAActionRankMemory(capacity=0, num_actions=4, device="cpu")
    memory.enqueue(torch.randn(2, 4), torch.zeros(2, 4))
    assert len(memory) == 0
    assert memory.snapshot() is None


def test_trainer_uses_cross_update_rank_memory_instead_of_clearing_each_update() -> None:
    from fate_oia.engine import train_tida_oia

    source = Path(train_tida_oia.__file__).read_text(encoding="utf-8")
    assert "action_rank_memory_capacity" in source
    assert 'action_rank_memory_reference", "video"' in source
    assert "rank_reference=rank_memory.snapshot()" in source
    assert 'output["image_action_logits"]' in source
    assert 'rank_memory.enqueue(rank_memory_logits, batch["action"])' in source
    assert "clear_rank_window(rank_window)" not in source
