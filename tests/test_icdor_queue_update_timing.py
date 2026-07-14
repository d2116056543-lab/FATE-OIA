from __future__ import annotations

import importlib

import torch


def test_accumulation_buffer_enqueues_once_only_after_successful_optimizer_step() -> None:
    module = importlib.import_module("fate_oia.optim.mosaic_soft_rank_queue")
    queue = module.MOSAICSoftRankQueue(2, capacity=16)
    pending = module.MOSAICAccumulationQueueBuffer()
    logits = torch.randn(2, 2, requires_grad=True)
    targets = torch.eye(2)
    pending.add(logits, targets, ["a", "b"])
    assert queue.count == 0
    assert all(not tensor.requires_grad for tensor in pending.tensor_payloads())
    pending.flush_after_optimizer_step(queue, optimizer_step_succeeded=True)
    assert queue.count == 2
    pending.flush_after_optimizer_step(queue, optimizer_step_succeeded=True)
    assert queue.count == 2
    assert queue.snapshot()["sample_hashes"].unique().numel() == 2

    restored = module.MOSAICSoftRankQueue(2, capacity=16)
    restored.load_state_dict(queue.state_dict())
    assert restored.count == queue.count
    assert torch.equal(restored.snapshot()["sample_hashes"], queue.snapshot()["sample_hashes"])

