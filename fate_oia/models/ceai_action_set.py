from __future__ import annotations

import torch
from torch import nn


DEFAULT_ACTION_PROTOTYPES = torch.tensor(
    [
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [0, 1, 1, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ],
    dtype=torch.float32,
)


class ActionSetPrototypeHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, prototype_vectors: torch.Tensor | None = None) -> None:
        super().__init__()
        proto = prototype_vectors.float() if prototype_vectors is not None else DEFAULT_ACTION_PROTOTYPES[:, :action_dim].clone()
        self.register_buffer("prototype_vectors", proto)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, proto.shape[0]))
        self.residual = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, action_dim))

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        summary = action_tokens.mean(dim=1)
        prototype_scores = self.score(summary)
        proto_logits = prototype_scores @ self.prototype_vectors.to(prototype_scores.dtype)
        residual = 0.05 * torch.tanh(self.residual(summary))
        logits = proto_logits + residual
        probs = torch.softmax(prototype_scores, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        stats = {
            "action_set_entropy": float(entropy.detach().cpu()),
            "action_set_residual_norm": float(residual.detach().norm(dim=-1).mean().cpu()),
            "action_set_top_proto": float(probs.detach().argmax(dim=-1).float().mean().cpu()),
        }
        return {"action_set_logits": logits, "prototype_scores": prototype_scores, "action_set_stats": stats}
