from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.acpr_pair_memory import ACPRPairMemory


def test_pair_memory_enqueue_uses_ring_buffer_not_full_cat():
    source = Path("fate_oia/models/acpr_pair_memory.py").read_text(encoding="utf-8")
    assert "torch.cat([old, payload[key]]" not in source
    assert "_memory_cursor" in source
    assert "index_copy_" in source


def test_pair_memory_ring_keeps_latest_samples():
    memory = ACPRPairMemory(memory_size=3, memory_device="cpu")
    for idx in range(5):
        memory.enqueue(
            [f"sample_{idx}"],
            torch.full((1, 4), float(idx)),
            torch.zeros(1, 2),
            torch.zeros(1, 4),
            torch.zeros(1, 3),
            contradiction_scores=torch.zeros(1, 3),
            reason_logits_detached=torch.zeros(1, 3),
            reason_embeddings_detached=torch.zeros(1, 3, 4),
        )
    assert memory._memory_count == 3
    assert memory._memory_cursor == 2
    names = memory._memory["file_names"]
    assert isinstance(names, list)
    valid = memory._valid_indices(torch.device("cpu")).tolist()
    kept = [names[i] for i in valid]
    assert kept == ["sample_2", "sample_3", "sample_4"]
