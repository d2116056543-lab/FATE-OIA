from __future__ import annotations

from typing import Sequence

import torch


def flatten_selected_predicates(
    history_tokens: torch.Tensor,
    terminal_predicate_tokens: torch.Tensor,
    *,
    predicate_indices: Sequence[int],
    action_count: int = 4,
) -> torch.Tensor:
    result = history_tokens.clone()
    for predicate_index in predicate_indices:
        query_index = action_count + int(predicate_index)
        result[:, :, query_index] = terminal_predicate_tokens[:, predicate_index : predicate_index + 1]
    return result


def apply_query_intervention(
    history_tokens: torch.Tensor,
    intervention: str,
    *,
    history_valid: torch.Tensor | None = None,
    terminal_predicate_tokens: torch.Tensor | None = None,
    predicate_indices: Sequence[int] | None = None,
    static_predicate_mask: torch.Tensor | None = None,
    action_count: int = 4,
) -> torch.Tensor:
    if intervention == "history_off":
        return torch.zeros_like(history_tokens)
    if intervention == "time_reverse":
        return history_tokens.flip(1)
    if intervention == "time_shuffle":
        generator = torch.Generator(device=history_tokens.device).manual_seed(20260821)
        return history_tokens[:, torch.randperm(history_tokens.shape[1], generator=generator, device=history_tokens.device)]
    if intervention == "repeated_last":
        if history_valid is None:
            index = torch.full((history_tokens.shape[0],), history_tokens.shape[1] - 1, device=history_tokens.device, dtype=torch.long)
        else:
            positions = torch.arange(history_tokens.shape[1], device=history_tokens.device)
            index = (positions[None] * history_valid.to(torch.long)).argmax(-1)
        terminal = history_tokens[torch.arange(history_tokens.shape[0], device=history_tokens.device), index]
        return terminal[:, None].expand_as(history_tokens)
    if intervention in ("selected_predicate_flatten", "matched_predicate_flatten"):
        if terminal_predicate_tokens is None or predicate_indices is None:
            raise ValueError(f"{intervention} requires terminal_predicate_tokens and predicate_indices")
        return flatten_selected_predicates(
            history_tokens, terminal_predicate_tokens, predicate_indices=predicate_indices, action_count=action_count
        )
    if intervention in ("static_only", "dynamic_only"):
        if static_predicate_mask is None:
            raise ValueError(f"{intervention} requires static_predicate_mask")
        result = history_tokens.clone()
        keep = static_predicate_mask if intervention == "static_only" else ~static_predicate_mask
        drop_indices = torch.where(~keep)[0] + action_count
        result[:, :, drop_indices] = 0
        if intervention == "static_only":
            result[:, :, :action_count] = 0
        return result
    if intervention == "wrong_action_route":
        result = history_tokens.clone()
        result[:, :, :action_count] = result[:, :, :action_count].roll(1, dims=2)
        return result
    raise ValueError(f"unknown temporal intervention: {intervention}")
