from __future__ import annotations

import torch
from torch import nn


class CastEvidenceGraph(nn.Module):
    def __init__(self, dim: int = 384, num_labels: int = 25, num_sets: int = 16, topk_edges: int = 64):
        super().__init__()
        self.dim = int(dim)
        self.num_labels = int(num_labels)
        self.num_sets = int(num_sets)
        self.topk_edges = int(topk_edges)
        self.edge_type_embedding = nn.Embedding(7, dim)
        self.edge_mlp = nn.Sequential(nn.Linear(dim * 6 + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.node_update = nn.Linear(dim * 2, dim)
        self.reason_to_set = nn.Linear(dim * 2, 1)

    def _edge_types(self, device: torch.device) -> torch.Tensor:
        n = self.num_labels + self.num_sets
        t = torch.zeros(n, n, dtype=torch.long, device=device)
        action = torch.arange(0, 4, device=device)
        reason = torch.arange(4, self.num_labels, device=device)
        sets = torch.arange(self.num_labels, n, device=device)
        t[action[:, None], action] = 0
        t[action[:, None], reason] = 1
        t[reason[:, None], action] = 2
        t[reason[:, None], reason] = 3
        t[reason[:, None], sets] = 4
        t[sets[:, None], reason] = 5
        t[sets[:, None], sets] = 6
        return t

    def forward(self, label_nodes, label_evidence, label_attention, action_set_nodes, text_similarity):
        b, l, d = label_nodes.shape
        nodes = torch.cat([label_nodes, action_set_nodes], dim=1)
        n = nodes.shape[1]
        set_evidence = action_set_nodes
        evidence = torch.cat([label_evidence, set_evidence], dim=1)
        ni = nodes.unsqueeze(2).expand(b, n, n, d)
        nj = nodes.unsqueeze(1).expand(b, n, n, d)
        ei = evidence.unsqueeze(2).expand(b, n, n, d)
        ej = evidence.unsqueeze(1).expand(b, n, n, d)
        overlap = torch.zeros(b, n, n, 1, device=nodes.device, dtype=nodes.dtype)
        attn_overlap = torch.einsum("bin,bjn->bij", label_attention, label_attention).unsqueeze(-1)
        overlap[:, :l, :l] = attn_overlap
        tsim = torch.zeros(n, n, device=nodes.device, dtype=nodes.dtype)
        tsim[:l, :l] = text_similarity.to(nodes.device, nodes.dtype)
        tsim = tsim.view(1, n, n, 1).expand(b, n, n, 1)
        edge_type_emb = self.edge_type_embedding(self._edge_types(nodes.device)).view(1, n, n, d).expand(b, n, n, d)
        ego_relation_summary = torch.zeros_like(overlap)
        feat = torch.cat([ni, nj, ei, ej, ei * ej, torch.abs(ei - ej) + edge_type_emb, overlap, tsim, ego_relation_summary], dim=-1)
        edge_logits = self.edge_mlp(feat).squeeze(-1)
        k = min(self.topk_edges, n)
        topv, topi = edge_logits.topk(k, dim=-1)
        masked = torch.full_like(edge_logits, -1e9)
        masked.scatter_(-1, topi, topv)
        edge_weights = torch.softmax(masked, dim=-1)
        msg = edge_weights @ nodes
        updated = nodes + self.node_update(torch.cat([nodes, msg], dim=-1))
        reason_nodes = updated[:, 4:l]
        set_nodes = updated[:, l:]
        r = reason_nodes.unsqueeze(2).expand(b, l - 4, self.num_sets, d)
        s = set_nodes.unsqueeze(1).expand(b, l - 4, self.num_sets, d)
        reason_to_set_logits = self.reason_to_set(torch.cat([r, s], dim=-1)).squeeze(-1)
        entropy = -(edge_weights.clamp_min(1e-9) * edge_weights.clamp_min(1e-9).log()).sum(-1).mean()
        r2s_mass = edge_weights[:, 4:l, l:].sum(-1).mean()
        return {
            "updated_label_nodes": updated[:, :l],
            "updated_set_nodes": set_nodes,
            "edge_logits": edge_logits,
            "edge_weights": edge_weights,
            "reason_to_set_logits": reason_to_set_logits,
            "graph_stats": {
                "graph_entropy": float(entropy.detach().cpu()),
                "reason_to_set_mass": float(r2s_mass.detach().cpu()),
            },
        }
