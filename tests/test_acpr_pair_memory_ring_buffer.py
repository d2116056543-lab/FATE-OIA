import inspect

import pytest
import torch

from fate_oia.models.acpr_pair_memory import ACPRPairMemory


def _payload(batch_size: int, offset: int = 0, device: str = "cpu"):
    file_names = [f"sample_{offset + i}" for i in range(batch_size)]
    global_embed = torch.randn(batch_size, 8, device=device)
    predicate_probs = torch.rand(batch_size, 5, device=device)
    action_targets = torch.zeros(batch_size, 4, device=device)
    reason_targets = torch.zeros(batch_size, 3, device=device)
    contradiction = torch.rand(batch_size, 3, device=device)
    reason_logits = torch.randn(batch_size, 3, device=device)
    reason_embeddings = torch.randn(batch_size, 3, 8, device=device)
    return file_names, global_embed, predicate_probs, action_targets, reason_targets, contradiction, reason_logits, reason_embeddings


def test_pair_memory_enqueue_does_not_rebuild_history_with_torch_cat():
    src = inspect.getsource(ACPRPairMemory.enqueue)
    assert "torch.cat" not in src
    assert "fixed-capacity ring buffer" in src


def test_pair_memory_ring_buffer_keeps_recent_samples_without_growing():
    mem = ACPRPairMemory(dim=8, memory_size=5, memory_device="cpu")
    for offset in (0, 2, 4):
        mem.enqueue(*_payload(2, offset=offset))

    view = mem.memory_view(torch.device("cpu"))
    assert mem.memory_count == 5
    assert view["global_embed"].shape[0] == 5
    assert view["reason_targets"].shape == (5, 3)
    assert view["file_names"] == ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]


def test_pair_memory_device_argument_controls_storage_device_cpu():
    mem = ACPRPairMemory(dim=8, memory_size=4, memory_device="cpu")
    mem.enqueue(*_payload(3, offset=0))
    view = mem.memory_view(torch.device("cpu"))
    for key in (
        "global_embed",
        "predicate_probs",
        "action_targets",
        "reason_targets",
        "contradiction_scores",
        "reason_logits_detached",
        "reason_embeddings_detached",
    ):
        assert view[key].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA storage check requires a GPU")
def test_pair_memory_device_argument_controls_storage_device_cuda():
    mem = ACPRPairMemory(dim=8, memory_size=4, memory_device="cuda")
    mem.enqueue(*_payload(3, offset=0, device="cuda"))
    view = mem.memory_view(torch.device("cuda"))
    assert view["global_embed"].device.type == "cuda"
    assert view["reason_embeddings_detached"].device.type == "cuda"
