from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from fate_oia.models.token_provenance import recover_attribution


def label_attention_map(attention: torch.Tensor, label_index: int, provenance: torch.Tensor | None = None) -> torch.Tensor:
    """Return label-conditioned token attribution from label-query attention."""
    if attention.ndim == 4:
        scores = attention.mean(1)[:, label_index]
    elif attention.ndim == 3:
        scores = attention[:, label_index]
    else:
        raise ValueError("attention must be [B,H,L,N] or [B,L,N]")
    if provenance is not None:
        scores = recover_attribution(scores, provenance)
    return _normalize(scores)


def gradient_x_attention(attention: torch.Tensor, attention_grad: torch.Tensor, label_index: int, provenance: torch.Tensor | None = None) -> torch.Tensor:
    if attention.shape != attention_grad.shape:
        raise ValueError("attention and attention_grad must have the same shape")
    positive_grad = F.relu(attention_grad)
    if attention.ndim == 4:
        scores = (attention * positive_grad).mean(1)[:, label_index]
    elif attention.ndim == 3:
        scores = (attention * positive_grad)[:, label_index]
    else:
        raise ValueError("attention must be [B,H,L,N] or [B,L,N]")
    if provenance is not None:
        scores = recover_attribution(scores, provenance)
    return _normalize(scores)


def snna_value_grad(
    attention: torch.Tensor,
    value_vectors: torch.Tensor,
    attention_grad: torch.Tensor,
    label_index: int,
    provenance: torch.Tensor | None = None,
) -> torch.Tensor:
    """SNNA-style attention x value-norm x positive label gradient."""
    if attention.shape != attention_grad.shape:
        raise ValueError("attention and attention_grad must have the same shape")
    if value_vectors.ndim == 4:
        value_norm = value_vectors.norm(dim=-1).mean(1)
    elif value_vectors.ndim == 3:
        value_norm = value_vectors.norm(dim=-1)
    else:
        raise ValueError("value_vectors must be [B,H,N,D] or [B,N,D]")
    if attention.ndim == 4:
        scores = (attention * F.relu(attention_grad)).mean(1)[:, label_index] * value_norm
    elif attention.ndim == 3:
        scores = (attention * F.relu(attention_grad))[:, label_index] * value_norm
    else:
        raise ValueError("attention must be [B,H,L,N] or [B,L,N]")
    if provenance is not None:
        scores = recover_attribution(scores, provenance)
    return _normalize(scores)


def smoothgrad_token_attribution(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    tokens: torch.Tensor,
    label_index: int,
    *,
    samples: int = 5,
    sigma: float = 0.15,
) -> torch.Tensor:
    """Gradient x activation SmoothGrad over token inputs for smoke/visualization."""
    if samples < 1:
        raise ValueError("samples must be >= 1")
    maps = []
    scale = float(tokens.detach().std().clamp_min(1e-6).item()) * float(sigma)
    for _ in range(samples):
        noisy = (tokens + torch.randn_like(tokens) * scale).detach().requires_grad_(True)
        logits = forward_fn(noisy)
        if logits.ndim != 2 or label_index >= logits.shape[1]:
            raise ValueError("forward_fn must return logits [B,L] containing label_index")
        score = logits[:, label_index].sum()
        grad = torch.autograd.grad(score, noisy, retain_graph=False, create_graph=False)[0]
        maps.append((grad * noisy).sum(-1).detach())
    return _normalize(torch.stack(maps, 0).mean(0))


def _normalize(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scores = scores.float()
    scores = scores - scores.amin(dim=-1, keepdim=True)
    denom = scores.amax(dim=-1, keepdim=True).clamp_min(eps)
    return scores / denom
