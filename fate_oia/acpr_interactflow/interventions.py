from __future__ import annotations

import torch

INTERVENTION_NAMES = [
    "global_only",
    "regime_off",
    "phase_off",
    "source_off",
    "factor_off",
    "predicate_off",
    "evidence_tube_off",
    "equal_mass_random",
    "temporal_reverse",
    "temporal_shuffle",
    "lag_disabled",
    "last_frame_only",
    "prefix_5",
    "prefix_10",
    "prefix_15",
]


def zero_predicate_intervention(predicate_probs: torch.Tensor, indices: list[int]) -> torch.Tensor:
    out = predicate_probs.clone()
    if indices:
        out[:, indices] = 0
    return out


def selected_vs_random_influence(action_logits: torch.Tensor, intervened_logits: torch.Tensor) -> torch.Tensor:
    return (action_logits.softmax(-1).max(-1).values - intervened_logits.softmax(-1).max(-1).values).detach()


def apply_temporal_intervention(frames: torch.Tensor, name: str) -> torch.Tensor:
    if name == "temporal_reverse":
        return frames.flip(1)
    if name == "temporal_shuffle":
        idx = torch.randperm(frames.shape[1], device=frames.device)
        return frames[:, idx]
    if name == "last_frame_only":
        return frames[:, -1:].expand_as(frames)
    if name == "prefix_5":
        out = frames.clone()
        out[:, 5:] = frames[:, 4:5]
        return out
    if name == "prefix_10":
        out = frames.clone()
        out[:, 10:] = frames[:, 9:10]
        return out
    if name == "prefix_15":
        return frames
    raise ValueError(f"unsupported temporal intervention: {name}")


def intervention_suite() -> list[str]:
    return list(INTERVENTION_NAMES)
