from __future__ import annotations

import torch
from torch import Tensor, nn


class CoEVEvidenceReadout(nn.Module):
    def __init__(self, dim: int = 384, basis_dim: int = 8) -> None:
        super().__init__()
        self.unary_basis = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, basis_dim))
        self.pair_basis = nn.Sequential(nn.Linear(14, 64), nn.GELU(), nn.Linear(64, basis_dim))
        self.dim = dim
        self.visual_weight = nn.Parameter(torch.empty(25, dim))
        self.bias = nn.Parameter(torch.zeros(25))
        self.coeff_action = nn.Parameter(torch.empty(4, 238, basis_dim))
        self.coeff_reason = nn.Parameter(torch.empty(21, 238, basis_dim))
        nn.init.normal_(self.coeff_action, std=0.01)
        nn.init.normal_(self.coeff_reason, std=0.01)
        nn.init.normal_(self.visual_weight, std=0.02)

    def forward(self, q_video: Tensor, lift: dict[str, Tensor]) -> dict[str, Tensor]:
        if q_video.shape[-2:] != (25, self.dim):
            raise ValueError("q_video must be [B,25,D]")
        with torch.autocast(device_type=q_video.device.type, enabled=False):
            unary = lift["unary"].float(); pair = lift["pair"].float()
            uv = lift["unary_valid"].float(); pv = lift["pair_valid"].float()
            ub = (self.unary_basis(unary) - self.unary_basis(torch.zeros_like(unary))) * uv.unsqueeze(-1)
            pb = (self.pair_basis(pair) - self.pair_basis(torch.zeros_like(pair))) * pv.unsqueeze(-1)
            basis = torch.cat((ub, pb), 1)
            coeff = torch.cat((self.coeff_action, self.coeff_reason), 0)
            contribution = torch.einsum("bfd,lfd->blf", basis, coeff)
            visual = torch.einsum("bld,ld->bl", q_video.float(), self.visual_weight)
            evidence = contribution.sum(-1)
            logits = self.bias + visual + evidence
        return {"logits": logits, "action_logits": logits[:, :4], "reason_logits": logits[:, 4:],
                "bias_logits": self.bias.unsqueeze(0).expand(q_video.shape[0], -1),
                "visual_logits": self.bias + visual, "evidence_logits": evidence,
                "factor_contribution": contribution}
