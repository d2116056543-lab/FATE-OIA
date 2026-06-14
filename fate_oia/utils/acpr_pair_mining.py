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
        "active_pair_count": int(pairs.get("active_pair_count", 0)),
        "hard_pair_count": int(pairs.get("hard_pair_count", pairs.get("hard_negative_count", 0))),
        "semi_hard_pair_count": int(pairs.get("semi_hard_pair_count", 0)),
        "easy_pair_count": int(pairs.get("easy_pair_count", 0)),
        "zero_loss_pair_count": int(pairs.get("zero_loss_pair_count", 0)),
        "margin_satisfied_count": int(pairs.get("margin_satisfied_count", 0)),
        "tail_active_pair_count": int(pairs.get("tail_active_pair_count", 0)),
        "pair_memory_count": int(pairs.get("pair_memory_count", 0)),
        "pair_no_candidate_count": int(pairs.get("pair_no_candidate_count", 0)),
        "pair_gate_filtered_count": int(pairs.get("pair_gate_filtered_count", 0)),
    }
    pair_count = max(int(out["pair_count"]), 1)
    out["active_pair_rate"] = float(out["active_pair_count"]) / pair_count
    out["matched_pair_zero_last100_proxy"] = int(out["zero_loss_pair_count"])
    for k in ["pair_action_sim", "pair_visual_sim", "pair_predicate_sim", "pair_contradiction", "pair_hinge_raw"]:
        v = pairs.get(k)
        out[k + "_mean"] = float(v.float().mean().detach().cpu()) if torch.is_tensor(v) and v.numel() else 0.0
    active = pairs.get("pair_active_mask")
    hinge = pairs.get("pair_hinge_raw")
    if torch.is_tensor(active) and torch.is_tensor(hinge) and active.numel() and hinge.numel():
        mask = active.bool()
        out["active_pair_hinge_mean"] = float(hinge[mask].float().mean().detach().cpu()) if mask.any() else 0.0
        out["pair_hinge_positive_rate"] = float((hinge.float() > 0).float().mean().detach().cpu())
    else:
        out["active_pair_hinge_mean"] = 0.0
        out["pair_hinge_positive_rate"] = 0.0
    return out
