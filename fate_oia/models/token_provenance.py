from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_provenance(batch: int, tokens: int, device=None, dtype=None) -> torch.Tensor:
    eye = torch.eye(tokens, device=device, dtype=dtype or torch.float32)
    return eye.unsqueeze(0).expand(batch, tokens, tokens).contiguous()


def keep_merge_tokens(tokens: torch.Tensor, scores: torch.Tensor | None = None, keep_ratio: float = 0.5, num_summary_tokens: int = 1, min_tokens: int = 1) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Keep top-k tokens and merge the rest into summary tokens.

    Returns:
        reduced_tokens: [B, K + M, D]
        provenance: [B, N, K + M], rows sum to 1 and map original tokens to reduced tokens.
        stats: metadata.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [B,N,D]")
    b, n, d = tokens.shape
    if scores is None:
        scores = tokens.norm(dim=-1)
    keep = max(min_tokens, int(round(n * keep_ratio)))
    keep = min(max(keep, 1), n)
    m = max(int(num_summary_tokens), 0)
    order = torch.argsort(scores, dim=1, descending=True)
    keep_idx = order[:, :keep]
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
    kept = torch.gather(tokens, 1, gather_idx)
    prov = torch.zeros(b, n, keep + m, device=tokens.device, dtype=tokens.dtype)
    batch_idx = torch.arange(b, device=tokens.device).unsqueeze(1)
    prov[batch_idx, keep_idx, torch.arange(keep, device=tokens.device).view(1, -1)] = 1.0
    if m > 0 and keep < n:
        mask = torch.ones(b, n, device=tokens.device, dtype=torch.bool)
        mask[batch_idx, keep_idx] = False
        rest = tokens.masked_fill(~mask.unsqueeze(-1), 0.0)
        denom = mask.sum(1).clamp_min(1).to(tokens.dtype).view(b, 1)
        summary = rest.sum(1, keepdim=True) / denom.view(b, 1, 1)
        if m > 1:
            summary = summary.expand(-1, m, -1).contiguous()
        prov[:, :, keep:] = mask.to(tokens.dtype).unsqueeze(-1) / float(m)
        reduced = torch.cat([kept, summary], dim=1)
    else:
        reduced = kept
        prov = prov[:, :, :keep]
    return reduced, prov, {"original_tokens": n, "kept_tokens": keep, "summary_tokens": m if keep < n else 0}


def recover_attribution(reduced_attr: torch.Tensor, provenance: torch.Tensor) -> torch.Tensor:
    if reduced_attr.ndim == 2:
        return torch.bmm(provenance, reduced_attr.unsqueeze(-1)).squeeze(-1)
    if reduced_attr.ndim == 3:
        return torch.bmm(provenance, reduced_attr)
    raise ValueError("reduced_attr must be [B,R] or [B,R,C]")
