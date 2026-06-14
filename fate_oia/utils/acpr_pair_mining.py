from __future__ import annotations

import torch


def action_vectors_to_subset_id(action: torch.Tensor) -> torch.Tensor:
    bits = torch.tensor([1, 2, 4, 8], device=action.device, dtype=torch.long)
    return ((action > 0.5).long() * bits.view(1, -1)).sum(-1)


def pair_summary(pairs: dict) -> dict[str, float | int]:
    out: dict[str, float | int] = {
        "pair_count": int(pairs.get("pair_count", 0)),
        "hard_negative_count": int(pairs.get("hard_negative_count", 0)),
        "tail_pair_count": int(pairs.get("tail_pair_count", 0)),
    }
    for k in ["pair_action_sim", "pair_visual_sim", "pair_predicate_sim", "pair_contradiction"]:
        v = pairs.get(k)
        out[k + "_mean"] = float(v.float().mean().detach().cpu()) if torch.is_tensor(v) and v.numel() else 0.0
    return out
