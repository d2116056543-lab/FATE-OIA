from __future__ import annotations

from typing import Any

import torch


def select_positive_reasons(
    reason_labels: torch.Tensor | None,
    reason_logits: torch.Tensor,
    tail_labels: tuple[int, ...],
    max_positive: int = 2,
) -> list[list[int]]:
    selected: list[list[int]] = []
    b, reason_dim = reason_logits.shape
    tail = [int(x) for x in tail_labels if 0 <= int(x) < reason_dim]
    for i in range(b):
        if reason_labels is not None:
            positives = [int(x) for x in torch.nonzero(reason_labels[i] > 0.5, as_tuple=False).flatten().tolist()]
            positives.sort(key=lambda r: (0 if r in tail else 1, r))
        else:
            positives = torch.topk(reason_logits[i], k=min(max_positive, reason_dim)).indices.tolist()
        selected.append(positives[: max(1, max_positive)])
    return selected


def target_unit_mask(evidence: dict[str, Any], selected: list[list[int]], allow_fallback: bool = False) -> torch.Tensor:
    ev_mask = evidence["evidence_mask"].clone()
    reason_mask = evidence.get("reason_evidence_mask")
    source_type = evidence.get("evidence_source_type")
    out = torch.zeros_like(ev_mask)
    if not isinstance(reason_mask, torch.Tensor):
        return out
    for b, reasons in enumerate(selected):
        for r in reasons:
            if 0 <= int(r) < reason_mask.shape[1]:
                out[b] |= reason_mask[b, int(r)]
    out &= ev_mask
    if not allow_fallback and isinstance(source_type, torch.Tensor):
        out &= source_type != 3
    return out


def make_evidence_override(
    evidence: dict[str, Any],
    target_mask: torch.Tensor,
    mode: str,
    null_token: torch.Tensor,
    replacement_tokens: torch.Tensor | None = None,
) -> dict[str, Any]:
    out = dict(evidence)
    tokens = evidence["evidence_tokens"].clone()
    mask = evidence["evidence_mask"].clone()
    quality = evidence["evidence_quality"].clone()
    null = null_token.view(1, 1, -1).to(tokens.device).to(tokens.dtype)
    if mode == "target_deleted":
        tokens = torch.where(target_mask.unsqueeze(-1), null.expand_as(tokens), tokens)
        quality = torch.where(target_mask, quality.new_zeros(()), quality)
        mask = mask & (~target_mask)
    elif mode == "context_only":
        keep = mask & (~target_mask)
        tokens = torch.where(keep.unsqueeze(-1), tokens, null.expand_as(tokens))
        quality = torch.where(keep, quality, quality.new_zeros(()))
        mask = keep
    elif mode == "evidence_only":
        keep = mask & target_mask
        tokens = torch.where(keep.unsqueeze(-1), tokens, null.expand_as(tokens))
        quality = torch.where(keep, quality, quality.new_zeros(()))
        mask = keep
    elif mode == "replaced":
        if replacement_tokens is None:
            replacement_tokens = tokens.roll(shifts=1, dims=0)
        tokens = torch.where(target_mask.unsqueeze(-1), replacement_tokens.to(tokens.device).to(tokens.dtype), tokens)
    elif mode == "factual":
        pass
    else:
        raise ValueError(f"Unknown counterfactual evidence mode: {mode}")
    out["evidence_tokens"] = tokens
    out["evidence_mask"] = mask
    out["evidence_quality"] = quality
    out["intervention_mode"] = mode
    out["target_unit_mask"] = target_mask
    return out
