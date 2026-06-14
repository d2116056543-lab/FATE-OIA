from __future__ import annotations

import torch


def action_vectors_to_subset_id(action: torch.Tensor) -> torch.Tensor:
    bits = torch.tensor([1, 2, 4, 8], device=action.device, dtype=torch.long)
    return ((action > 0.5).long() * bits.view(1, -1)).sum(-1)


def pair_summary(pairs: dict) -> dict[str, int]:
    return {
        "positive_pair_count": int(pairs.get("positive_pairs").shape[0]) if pairs.get("positive_pairs") is not None else 0,
        "contrast_pair_count": int(pairs.get("contrast_pairs").shape[0]) if pairs.get("contrast_pairs") is not None else 0,
        "tail_pair_count": int(pairs.get("tail_pair_count", 0)),
    }
