from __future__ import annotations

import torch
import torch.nn.functional as F


def identity_provenance(batch: int, tokens: int, device=None, dtype=None) -> torch.Tensor:
    eye = torch.eye(tokens, device=device, dtype=dtype or torch.float32)
    return eye.unsqueeze(0).expand(batch, tokens, tokens).contiguous()


def _cluster_assign(rest_tokens: torch.Tensor, num_clusters: int) -> torch.Tensor:
    """Deterministic farthest-first cluster assignment for one sample."""
    n = rest_tokens.shape[0]
    if num_clusters <= 1 or n <= 1:
        return torch.zeros(n, device=rest_tokens.device, dtype=torch.long)
    k = min(num_clusters, n)
    normed = F.normalize(rest_tokens.float(), dim=-1)
    centers = [0]
    dist = 1.0 - (normed @ normed[0].view(-1, 1)).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(dist).item())
        centers.append(idx)
        new_dist = 1.0 - (normed @ normed[idx].view(-1, 1)).squeeze(1)
        dist = torch.minimum(dist, new_dist)
    center_tensor = torch.tensor(centers, device=rest_tokens.device, dtype=torch.long)
    sim = normed @ normed[center_tensor].transpose(0, 1)
    assign = torch.argmax(sim, dim=1)
    return assign


def cluster_merge_tokens(rest_tokens: torch.Tensor, num_summary_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster residual tokens into non-duplicate summary tokens.

    Returns summary tokens [B,M,D] and residual-to-summary assignment weights
    [B,R,M] whose rows sum to 1.
    """
    b, r, d = rest_tokens.shape
    m = max(1, int(num_summary_tokens))
    summaries = []
    weights = rest_tokens.new_zeros(b, r, m)
    for bi in range(b):
        assign = _cluster_assign(rest_tokens[bi], m)
        sample_summaries = []
        for ci in range(m):
            mask = assign == ci
            if not bool(mask.any()):
                sample_summaries.append(rest_tokens[bi].mean(0))
                continue
            sample_summaries.append(rest_tokens[bi, mask].mean(0))
            weights[bi, mask, ci] = 1.0
        summaries.append(torch.stack(sample_summaries, dim=0))
    return torch.stack(summaries, dim=0), weights


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
    keep_idx = torch.sort(order[:, :keep], dim=1).values
    gather_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)
    kept = torch.gather(tokens, 1, gather_idx)
    prov = torch.zeros(b, n, keep + m, device=tokens.device, dtype=tokens.dtype)
    batch_idx = torch.arange(b, device=tokens.device).unsqueeze(1)
    prov[batch_idx, keep_idx, torch.arange(keep, device=tokens.device).view(1, -1)] = 1.0
    if m > 0 and keep < n:
        mask = torch.ones(b, n, device=tokens.device, dtype=torch.bool)
        mask[batch_idx, keep_idx] = False
        rest_rows = []
        rest_to_original = []
        for bi in range(b):
            idx = torch.nonzero(mask[bi], as_tuple=False).squeeze(1)
            rest_rows.append(tokens[bi, idx])
            rest_to_original.append(idx)
        max_rest = max(x.shape[0] for x in rest_rows)
        padded = tokens.new_zeros(b, max_rest, d)
        valid = torch.zeros(b, max_rest, device=tokens.device, dtype=torch.bool)
        for bi, row in enumerate(rest_rows):
            padded[bi, : row.shape[0]] = row
            valid[bi, : row.shape[0]] = True
        summary, weights = cluster_merge_tokens(padded, m)
        for bi, idx in enumerate(rest_to_original):
            prov[bi, idx, keep:] = weights[bi, : idx.shape[0]]
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
