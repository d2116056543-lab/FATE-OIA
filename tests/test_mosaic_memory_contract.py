from __future__ import annotations

import inspect

import torch

from fate_oia.models.mosaic_sparse_label_decoder import MOSAICSparseLabelDecoder
from fate_oia.optim.mosaic_soft_rank_queue import MOSAICSoftRankQueue
from fate_oia.engine import profile_acpr_mosaic_ad


def test_soft_rank_queue_is_fixed_storage_ring_without_cat() -> None:
    source = inspect.getsource(MOSAICSoftRankQueue.enqueue)
    assert "torch.cat" not in source
    queue = MOSAICSoftRankQueue(3, capacity=5)
    pointers = (queue.logit_buffer.data_ptr(), queue.target_buffer.data_ptr())
    for step in range(4):
        queue.enqueue(torch.full((2, 3), float(step)), torch.zeros(2, 3), [f"{step}-a", f"{step}-b"])
        assert (queue.logit_buffer.data_ptr(), queue.target_buffer.data_ptr()) == pointers
    assert queue.count == 5
    assert queue.snapshot()["logits"].shape == (5, 3)


def test_queue_checkpoint_restores_python_ring_counters() -> None:
    queue = MOSAICSoftRankQueue(2, capacity=4)
    queue.enqueue(torch.ones(3, 2), torch.zeros(3, 2), ["a", "b", "c"])
    restored = MOSAICSoftRankQueue(2, capacity=4)
    restored.load_state_dict(queue.state_dict())
    assert restored.count == 3
    restored.enqueue(torch.ones(2, 2), torch.zeros(2, 2), ["d", "e"])
    assert restored.count == 4
    assert restored.snapshot()["logits"].shape == (4, 2)


def test_sparse_decoder_retrieval_budget_never_materializes_dense_factor_token_features() -> None:
    decoder = MOSAICSparseLabelDecoder(
        4, dim=16, decoder_layers=1, self_attention_heads=4, highres_topk=32, midres_topk=16
    )
    pyramid = {
        "F_hi": torch.randn(2, 16, 45, 80),
        "F_mid": torch.randn(2, 16, 23, 40),
        "F_ctx": torch.randn(2, 16, 12, 20),
    }
    output = decoder(pyramid)
    assert output["highres_indices"].shape == (2, 4, 32)
    assert output["midres_indices"].shape == (2, 4, 16)
    assert output["retrieval_attention"].shape == (2, 4, 48)
    assert all(tensor.numel() < 2 * 4 * 3600 * 16 for tensor in output.values() if isinstance(tensor, torch.Tensor))


def test_runtime_profiler_measures_loader_and_cuda_timing_after_warmup() -> None:
    run_source = inspect.getsource(profile_acpr_mosaic_ad._run_steps)
    profile_source = inspect.getsource(profile_acpr_mosaic_ad.profile)
    assert "profile_timing=True" in run_source
    assert "dataloader_load_time_sec" in run_source
    assert "device_step_time_sec" in run_source
    assert "dataloader_stalls" in run_source
    assert "max_dataloader_load_sec" in run_source
    assert "dataloader_stall_steps" in run_source
    assert "median_step_sec" in run_source and "p95_step_sec" in run_source
    assert "median_step_ms" in run_source and "p95_step_ms" in run_source
    assert "cuda_retries" in run_source and "nan_count" in run_source
    # Candidate timing and the final stability probe must both exclude loader startup.
    assert profile_source.count("warmup_steps=warmup_steps") >= 2
    assert "write_json(output, failure_payload)" in profile_source
    train_source = inspect.getsource(__import__("fate_oia.engine.train_acpr_mosaic_ad", fromlist=["train_representation_epoch"]).train_representation_epoch)
    assert "torch.cuda.Event" in train_source
