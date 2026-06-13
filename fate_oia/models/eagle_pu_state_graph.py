from __future__ import annotations

import torch
from torch import nn

from .eagle_pu_sparse_ops import entmax15_bisect


class StateGroundedLabelGraph(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, use_action_delta: bool = False) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.num_sets = 16
        self.num_nodes = self.num_labels + self.num_sets
        self.use_action_delta = use_action_delta
        self.set_nodes = nn.Parameter(torch.randn(self.num_sets, dim) * 0.02)
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 4 + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.reason_to_set = nn.Linear(dim, self.num_sets)
        self.reason_delta = nn.Linear(dim, 1)
        self.action_delta = nn.Linear(dim, 1)

    def forward(self, label_nodes: torch.Tensor, state_tokens: torch.Tensor, text_similarity: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict[str, float]]:
        b, l, d = label_nodes.shape
        set_nodes = self.set_nodes.unsqueeze(0).expand(b, -1, -1)
        nodes = torch.cat([label_nodes, set_nodes], dim=1)
        ni = nodes.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        nj = nodes.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        prod = ni * nj
        diff = (ni - nj).abs()
        overlap = torch.sigmoid((ni * nj).mean(-1, keepdim=True))
        if text_similarity is None:
            text = torch.zeros(b, self.num_nodes, self.num_nodes, 1, device=nodes.device, dtype=nodes.dtype)
        else:
            sim = torch.zeros(self.num_nodes, self.num_nodes, device=nodes.device, dtype=nodes.dtype)
            sim[:l, :l] = text_similarity.to(nodes.device, nodes.dtype)
            text = sim.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, -1)
        ego_relation = torch.sigmoid(state_tokens.mean(1).mean(-1, keepdim=True)).view(b, 1, 1, 1).expand(-1, self.num_nodes, self.num_nodes, -1)
        edge_input = torch.cat([ni, nj, prod, diff, overlap, text, ego_relation], dim=-1)
        logits = self.edge_mlp(edge_input).squeeze(-1)
        mask = torch.eye(self.num_nodes, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        logits = logits.masked_fill(mask, -1e4)
        edge_weights = entmax15_bisect(logits, dim=-1)
        message = torch.bmm(edge_weights, nodes)
        reason_message = message[:, self.action_dim : self.action_dim + self.reason_dim]
        action_message = message[:, : self.action_dim]
        reason_graph_delta = 0.08 * torch.tanh(self.reason_delta(reason_message).squeeze(-1) / 0.08)
        if self.use_action_delta:
            action_graph_delta = 0.02 * torch.tanh(self.action_delta(action_message).squeeze(-1) / 0.02)
        else:
            action_graph_delta = torch.zeros(b, self.action_dim, device=nodes.device, dtype=nodes.dtype)
        reason_to_set_logits = self.reason_to_set(reason_message)
        entropy = (-(edge_weights.clamp_min(1e-8).log() * edge_weights).sum(-1)).mean()
        return {"edge_weights": edge_weights, "reason_to_set_logits": reason_to_set_logits, "reason_graph_delta": reason_graph_delta, "action_graph_delta": action_graph_delta, "state_graph_stats": {"graph_entropy": float(entropy.detach().cpu())}}
