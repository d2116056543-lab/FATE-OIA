from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from fate_oia.metrics import multilabel_metrics_from_logits


def branch_metrics(action_logits: Tensor, reason_logits: Tensor, action_target: Tensor,
                   reason_target: Tensor, threshold: float | Tensor = 0.5) -> dict[str, Any]:
    at, rt = threshold, threshold
    if isinstance(threshold, Tensor) and threshold.numel() == 25:
        at, rt = threshold[:4], threshold[4:]
    action = multilabel_metrics_from_logits(action_logits.float(), action_target.float(), threshold=at, prefix="Act_")
    reason = multilabel_metrics_from_logits(reason_logits.float(), reason_target.float(), threshold=rt, prefix="Exp_")
    return {**action, **reason, "joint": 0.5 * action["Act_mF1"] + 0.5 * reason["Exp_mF1"]}


def evidence_diagnostics(output: dict[str, Tensor]) -> dict[str, float]:
    def rms(value: Tensor) -> float:
        return float(value.float().square().mean().sqrt())

    def entropy(value: Tensor) -> Tensor:
        value = value.float().clamp_min(1e-9)
        return -(value * value.log()).sum(-1)

    def quantiles(value: Tensor, prefix: str) -> dict[str, float]:
        flat = value.float().flatten()
        points = torch.quantile(flat, flat.new_tensor([0.1, 0.5, 0.9]))
        return {f"{prefix}_p10": float(points[0]), f"{prefix}_p50": float(points[1]),
                f"{prefix}_p90": float(points[2])}

    mixture = output["predicate_mixture"].float()
    active = mixture.sum(-1) > 0
    entropy = -(mixture.clamp_min(1e-9) * mixture.clamp_min(1e-9).log()).sum(-1)
    transport_token = (output["atom_token"] - output["atom_token_pre_transport"]).square().mean().sqrt()
    transport_map = (output["atom_map"] - output["atom_map_pre_transport"]).square().mean().sqrt()
    maps = output["atom_map"].float()
    norm = maps / maps.sum(-1, keepdim=True).clamp_min(1e-9)
    overlap = torch.einsum("bakn,baln->bakl", norm, norm)
    eye = torch.eye(overlap.shape[-1], device=overlap.device, dtype=torch.bool)[None, None]
    offdiag = overlap.masked_select(~eye.expand_as(overlap))
    contribution = output["bounded_contribution"].float()
    budget = output["reason_budget"].float()
    reason_delta = output["reason_delta"].float()
    result = {"predicate_mixture_active_rate": float(active.float().mean()),
        "predicate_fallback_rate": float((~active).float().mean()),
        "predicate_effective_count": float(entropy.exp().mean()),
        "predicate_top1_mass": float(mixture.amax(-1).mean()),
        "predicate_prior_strength_mean": float(output["predicate_prior_strength"].float().mean()),
        "global_map_entropy": float((-(output["global_attention"].float().clamp_min(1e-9) *
                                      output["global_attention"].float().clamp_min(1e-9).log()).sum(-1)).mean()),
        "pre_transport_map_entropy": float((-(output["atom_map_pre_transport"].float().clamp_min(1e-9) *
                                             output["atom_map_pre_transport"].float().clamp_min(1e-9).log()).sum(-1)).mean()),
        "post_transport_map_entropy": float((-(norm.clamp_min(1e-9) * norm.clamp_min(1e-9).log()).sum(-1)).mean()),
        "atom_overlap_mean": float(offdiag.mean()), "atom_overlap_p90": float(torch.quantile(offdiag, 0.9)),
        "atom_overlap_over_ceiling_rate": float((offdiag > 0.2).float().mean()),
        "transport_token_delta_rms": float(transport_token), "transport_map_delta_rms": float(transport_map),
        "transport_gamma_mean": float(output["atom_transport_gamma"].float().mean()),
        "transport_offdiag_mass": float((output["atom_transport_matrix"].float() *
            (~torch.eye(output["atom_transport_matrix"].shape[-1], device=maps.device, dtype=torch.bool))[None,None]).sum(-1).mean()),
        "cotransport_matrix_discrepancy": 0.0,
        "local_global_token_rms_ratio": rms(output["local_token"]) / max(rms(output["global_token"]), 1e-9),
        "offset_rms": rms(output["sampling_offsets"]),
        "offset_max": float(output["sampling_offsets"].float().abs().amax()),
        "background_token_rms": rms(output["background_token"]),
        "centered_raw_token_norm_ratio": rms(output["centered_atom_token"]) / max(rms(output["atom_token"]), 1e-9),
        "contribution_reconstruction_error": float(output["contribution_reconstruction_error"]),
        "action_primary_logit_rms": rms(output["action_logits_primary"]),
        "action_final_logit_rms": rms(output["action_logits_final"]),
        "per_action_contribution_rms": contribution.square().mean((0, 2)).sqrt().tolist(),
        "contribution_rms": rms(contribution), "contribution_positive_rate": float((contribution > 0).float().mean()),
        "contribution_negative_rate": float((contribution < 0).float().mean()),
        "action_delta_rms": rms(output["action_delta"]), "action_cap_rate": float((output["action_delta"].abs() >= 0.999).float().mean()),
        "reason_budget_mean": float(budget.mean()), "reason_budget_min": float(budget.amin()), "reason_budget_max": float(budget.amax()),
        "reason_delta_rms": rms(reason_delta),
        "reason_delta_to_budget_ratio": float((reason_delta.abs() / budget.clamp_min(1e-6)).mean()),
        "naming_quality_mean": float(output["name_quality"].float().mean()),
        "named_coverage": float(output["named_coverage"].float()),
    }
    result.update(quantiles(output["action_delta"], "action_delta"))
    result.update(quantiles(budget, "reason_budget"))
    result.update(quantiles(reason_delta, "reason_delta"))
    return result
