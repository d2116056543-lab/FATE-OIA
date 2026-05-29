from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


class ActionSetHead(nn.Module):
    """Exact action-vector specialist head.

    It predicts frequent 4-bit action patterns and converts the pattern
    distribution back into marginal action logits.
    """

    def __init__(self, dim: int = 384, reason_dim: int = 21, action_dim: int = 4, pattern_matrix: torch.Tensor | None = None, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        if pattern_matrix is None:
            pattern_matrix = default_action_patterns(action_dim)
        self.register_buffer("pattern_matrix", pattern_matrix.float())
        num_patterns = int(pattern_matrix.shape[0])
        self.net = nn.Sequential(
            nn.LayerNorm(dim + action_dim + reason_dim),
            nn.Linear(dim + action_dim + reason_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_patterns),
        )

    def forward(self, base_action_logits: torch.Tensor, final_reason_logits: torch.Tensor, label_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = label_tokens.mean(1)
        features = torch.cat([pooled, base_action_logits, final_reason_logits], dim=-1)
        pattern_logits = self.net(features)
        probs = torch.softmax(pattern_logits, dim=-1)
        pattern_probs = (probs @ self.pattern_matrix.to(device=probs.device, dtype=probs.dtype)).clamp(1e-4, 1.0 - 1e-4)
        pattern_action_logits = torch.logit(pattern_probs)
        return pattern_logits, pattern_action_logits


def default_action_patterns(action_dim: int = 4) -> torch.Tensor:
    patterns = []
    for i in range(2 ** int(action_dim)):
        patterns.append([(i >> bit) & 1 for bit in range(action_dim)])
    return torch.tensor(patterns, dtype=torch.float32)


def build_action_patterns(actions: torch.Tensor, top_k: int = 16) -> tuple[torch.Tensor, dict[str, int | list]]:
    if actions.ndim != 2:
        raise ValueError("actions must be [N,A]")
    action_dim = actions.shape[1]
    binary = (actions[:, :action_dim] > 0.5).int()
    unique, counts = torch.unique(binary, dim=0, return_counts=True)
    order = torch.argsort(counts, descending=True)
    unique = unique[order]
    counts = counts[order]
    keep = min(int(top_k), unique.shape[0])
    kept = unique[:keep].float()
    if unique.shape[0] > keep:
        other_counts = counts[keep:].float().view(-1, 1)
        other = (unique[keep:].float() * other_counts).sum(0) / other_counts.sum().clamp_min(1.0)
        matrix = torch.cat([kept, other.view(1, -1)], dim=0)
    else:
        matrix = kept
    meta = {"num_patterns": int(matrix.shape[0]), "kept_counts": counts[:keep].cpu().int().tolist(), "patterns": matrix.cpu().tolist()}
    return matrix, meta


def assign_action_patterns(actions: torch.Tensor, pattern_matrix: torch.Tensor) -> torch.Tensor:
    binary = (actions > 0.5).float()
    hard_patterns = (pattern_matrix > 0.5).float()
    dist = (binary.unsqueeze(1) - hard_patterns.unsqueeze(0)).abs().sum(-1)
    return torch.argmin(dist, dim=1)


def action_set_accuracy(pattern_logits: torch.Tensor, pattern_ids: torch.Tensor) -> dict[str, float]:
    pred = pattern_logits.argmax(-1)
    topk = torch.topk(pattern_logits, k=min(3, pattern_logits.shape[1]), dim=-1).indices
    return {
        "action_set_acc": float((pred == pattern_ids).float().mean().item()),
        "action_set_top3_acc": float((topk == pattern_ids.view(-1, 1)).any(1).float().mean().item()),
    }
