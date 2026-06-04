from __future__ import annotations

import torch
from torch import nn


class ExpertBlock(nn.Module):
    def __init__(self, dim: int, ffn_ratio: float = 2.0, dropout: float = 0.05) -> None:
        super().__init__()
        hidden = int(dim * ffn_ratio)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ProgressiveLayeredExperts(nn.Module):
    """Two-stage PLE with distinct shared/action/reason/tail experts."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, tail_indices: tuple[int, ...] = (5, 6, 9, 10, 11, 12, 13, 14)) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.tail_indices = tuple(int(i) for i in tail_indices)
        self.shared_1 = ExpertBlock(dim)
        self.action_1 = ExpertBlock(dim)
        self.reason_1 = ExpertBlock(dim)
        self.shared_2 = ExpertBlock(dim)
        self.action_2 = ExpertBlock(dim)
        self.reason_2 = ExpertBlock(dim)
        self.tail_2 = ExpertBlock(dim)
        self.action_gate_1 = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=-1))
        self.reason_gate_1 = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=-1))
        self.action_gate_2 = nn.Sequential(nn.Linear(dim, 3), nn.Softmax(dim=-1))
        self.reason_gate_2 = nn.Sequential(nn.Linear(dim, 4), nn.Softmax(dim=-1))

    def _mix(self, weights: torch.Tensor, parts: list[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(parts, dim=2)
        return (stacked * weights.unsqueeze(1).unsqueeze(-1)).sum(dim=2)

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, shared_context: torch.Tensor) -> dict[str, torch.Tensor]:
        shared_tokens = torch.cat([action_tokens, reason_tokens], dim=1)
        s1_all = self.shared_1(shared_tokens)
        s1_action = s1_all[:, : self.action_dim]
        s1_reason = s1_all[:, self.action_dim :]
        a1 = self.action_1(action_tokens)
        r1 = self.reason_1(reason_tokens)
        wg_a1 = self.action_gate_1(shared_context)
        wg_r1 = self.reason_gate_1(shared_context)
        action_stage1 = self._mix(wg_a1, [s1_action, a1, action_tokens])
        reason_stage1 = self._mix(wg_r1, [s1_reason, r1, reason_tokens])

        s2_all = self.shared_2(torch.cat([action_stage1, reason_stage1], dim=1))
        s2_action = s2_all[:, : self.action_dim]
        s2_reason = s2_all[:, self.action_dim :]
        a2 = self.action_2(action_stage1)
        r2 = self.reason_2(reason_stage1)
        tail = self.tail_2(reason_stage1)
        tail_mask = torch.zeros(1, self.reason_dim, 1, device=reason_tokens.device, dtype=reason_tokens.dtype)
        if self.tail_indices:
            tail_mask[:, list(self.tail_indices), :] = 1.0
        tail_reason = tail * tail_mask + reason_stage1 * (1.0 - tail_mask)
        wg_a2 = self.action_gate_2(shared_context)
        wg_r2 = self.reason_gate_2(shared_context)
        action_out = self._mix(wg_a2, [s2_action, a2, action_stage1])
        reason_out = self._mix(wg_r2, [s2_reason, r2, tail_reason, reason_stage1])
        return {
            "action_tokens": action_out,
            "reason_tokens": reason_out,
            "tail_reason_tokens": tail_reason,
            "tail_reason_gate": tail_mask.expand(reason_tokens.shape[0], -1, -1),
            "action_gate_stage1": wg_a1,
            "reason_gate_stage1": wg_r1,
            "action_gate_stage2": wg_a2,
            "reason_gate_stage2": wg_r2,
        }
