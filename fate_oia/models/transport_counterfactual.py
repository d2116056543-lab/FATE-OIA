from __future__ import annotations

from typing import Any

import torch


def _selected(labels: torch.Tensor | None, logits: torch.Tensor, tail_labels: tuple[int, ...], max_positive: int) -> list[list[int]]:
    out = []
    probs = torch.sigmoid(logits.detach())
    for b in range(logits.shape[0]):
        chosen: list[int] = []
        if labels is not None:
            pos = [int(i) for i in torch.nonzero(labels[b] > 0.5, as_tuple=False).flatten().tolist()]
            chosen += [i for i in pos if i in tail_labels][:max_positive]
            chosen += [i for i in pos if i not in chosen][: max(0, max_positive - len(chosen))]
        if not chosen:
            chosen = [int(i) for i in probs[b].topk(k=min(max_positive, probs.shape[1])).indices.tolist()]
        out.append(chosen[:max_positive])
    return out


def _renorm(T: torch.Tensor) -> torch.Tensor:
    return T / T.sum(dim=(2, 3), keepdim=True).clamp_min(1e-8)


def _final(base: torch.Tensor, ev: torch.Tensor, alpha: torch.Tensor | float, clip: float) -> torch.Tensor:
    if not torch.is_tensor(alpha):
        alpha = torch.tensor(float(alpha), device=base.device, dtype=base.dtype)
    return base + alpha * (ev - base.detach()).clamp(-float(clip), float(clip))


def transport_counterfactual_intervention(
    base_reason_logits: torch.Tensor,
    evidence_reason_logits: torch.Tensor,
    transport_out: dict[str, Any],
    transport_module: Any,
    reason_labels: torch.Tensor | None,
    tail_labels: tuple[int, ...] = (12, 9, 5, 14, 6, 11, 10, 13),
    alpha: torch.Tensor | float = 0.08,
    clip_values: float = 1.0,
    max_positive_reasons_per_sample: int = 2,
    top_mass_fraction: float = 0.80,
) -> dict[str, Any]:
    T, sim, source_type = transport_out["T"], transport_out["sim"], transport_out["evidence_source_type"]
    mask = torch.zeros_like(T, dtype=torch.bool)
    for b, reasons in enumerate(_selected(reason_labels, base_reason_logits, tail_labels, max_positive_reasons_per_sample)):
        for r in reasons:
            flat = T[b, r].flatten()
            vals, idx = torch.sort(flat, descending=True)
            n = int(((vals.cumsum(0) / flat.sum().clamp_min(1e-8)) <= float(top_mass_fraction)).sum().item()) + 1
            mask[b, r].view(-1)[idx[: max(1, min(n, idx.numel()))]] = True
    deleted_T = _renorm(T.masked_fill(mask, 0.0))
    only_T = torch.zeros_like(T)
    only_T[mask] = T[mask]
    only_T = _renorm(only_T)
    deleted = _final(base_reason_logits, transport_module.recompute_from_T(deleted_T, sim, source_type)["evidence_reason_logits"], alpha, clip_values)
    only = _final(base_reason_logits, transport_module.recompute_from_T(only_T, sim, source_type)["evidence_reason_logits"], alpha, clip_values)
    factual = _final(base_reason_logits, evidence_reason_logits, alpha, clip_values)
    valid = mask.any(dim=(2, 3))
    non = ~valid
    return {
        "reason_logits_factual": factual,
        "reason_logits_target_deleted": deleted,
        "reason_logits_context_only": deleted,
        "reason_logits_evidence_only": only,
        "reason_logits_replaced": deleted,
        "cf_valid_mask": valid,
        "cf_selected_reason_indices": _selected(reason_labels, base_reason_logits, tail_labels, max_positive_reasons_per_sample),
        "cf_target_transport_mask": mask,
        "cf_target_evidence_count": mask.any(dim=2).sum(-1).to(base_reason_logits.dtype),
        "cf_transport_mass_removed": (T * mask.to(T.dtype)).sum(dim=(2, 3)),
        "target_deleted_drop_mean": (factual - deleted).masked_select(valid).mean() if valid.any() else factual.new_zeros(()),
        "non_target_deleted_drop_mean": (factual - deleted).masked_select(non).mean() if non.any() else factual.new_zeros(()),
        "cf_is_proxy": False,
        "intervention_type": "transport_mass_do_delete",
    }
