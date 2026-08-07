from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AIECertAtomTransport(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, atoms: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        init = math.log(0.05 / (0.25 - 0.05))
        self.gamma_raw = nn.Parameter(torch.full((action_dim,), init))

    def forward(self, token_pre: Tensor, map_pre: Tensor, gamma_cap: float = 0.25) -> dict[str, Tensor]:
        b, actions, atoms, dim = token_pre.shape
        flat = token_pre.reshape(b * actions, atoms, dim)
        _, heads = self.attention(flat, flat, flat, need_weights=True, average_attn_weights=False)
        matrix = heads.mean(1)
        eye = torch.eye(atoms, device=matrix.device, dtype=matrix.dtype)[None]
        matrix = matrix * (1.0 - eye)
        matrix = matrix / matrix.sum(-1, keepdim=True).clamp_min(1e-8)
        matrix = matrix.reshape(b, actions, atoms, atoms)
        learned_gamma = 0.25 * torch.sigmoid(self.gamma_raw).view(1, actions, 1, 1)
        gamma = torch.minimum(learned_gamma, learned_gamma.new_tensor(float(gamma_cap)))
        token_delta = torch.einsum("bakj,bajd->bakd", matrix, token_pre)
        map_delta = torch.einsum("bakj,bajn->bakn", matrix, map_pre)
        token = self.norm(token_pre + gamma * token_delta)
        atom_map = (map_pre + gamma * map_delta).clamp_min(0.0)
        atom_map = atom_map / atom_map.sum(-1, keepdim=True).clamp_min(1e-8)
        return {
            "atom_transport_matrix": matrix,
            "atom_transport_gamma": gamma.squeeze(-1).squeeze(-1).mean(0),
            "atom_token": token,
            "atom_map": atom_map,
        }
