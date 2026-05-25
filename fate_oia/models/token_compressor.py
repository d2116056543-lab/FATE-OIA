from __future__ import annotations

import torch

from fate_oia.models.token_provenance import identity_provenance, keep_merge_tokens, recover_attribution


def compute_token_scores(
    tokens: torch.Tensor,
    *,
    label_attention: torch.Tensor | None = None,
    grounding_prior: torch.Tensor | None = None,
    uncertainty: torch.Tensor | None = None,
    mode: str = "norm",
    alpha_label: float = 1.0,
    beta_grounding: float = 0.5,
    gamma_uncertainty: float = 0.2,
    delta_norm: float = 0.1,
) -> torch.Tensor:
    """Compute rationale-aware token scores for compression.

    ``norm`` remains the fallback, but hybrid scoring can combine label-query
    attention, downsampled grounding priors, uncertainty and token norm. All
    inputs are expected to be [B,N] except tokens [B,N,D].
    """
    norm_score = tokens.norm(dim=-1)
    if mode == "norm":
        return norm_score
    score = torch.zeros_like(norm_score)
    if mode in {"label_attention", "hybrid"} and label_attention is not None:
        score = score + float(alpha_label) * label_attention.to(score.device, score.dtype)
    if mode in {"grounding_prior", "hybrid"} and grounding_prior is not None:
        score = score + float(beta_grounding) * grounding_prior.to(score.device, score.dtype)
    if mode == "hybrid" and uncertainty is not None:
        score = score + float(gamma_uncertainty) * uncertainty.to(score.device, score.dtype)
    if mode == "hybrid":
        score = score + float(delta_norm) * norm_score
    if mode in {"label_attention", "grounding_prior"} and torch.all(score == 0):
        return norm_score
    return score


def compression_keep_ratio_for_epoch(
    epoch: int,
    *,
    start_epoch: int,
    keep_ratio_start: float,
    keep_ratio_final: float,
    warmup_epochs: int,
) -> float:
    if epoch < start_epoch:
        return 1.0
    if warmup_epochs <= 0:
        return float(keep_ratio_final)
    t = min(max(epoch - start_epoch, 0), warmup_epochs) / float(warmup_epochs)
    return float(keep_ratio_start + t * (keep_ratio_final - keep_ratio_start))


__all__ = [
    "identity_provenance",
    "keep_merge_tokens",
    "recover_attribution",
    "compute_token_scores",
    "compression_keep_ratio_for_epoch",
]
