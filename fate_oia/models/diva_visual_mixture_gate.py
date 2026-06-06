from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SupervisedVisualMixtureGate(nn.Module):
    def __init__(self, action_dim: int = 4, hidden_dim: int = 64, delta_cap: float = 0.08, gate_margin: float = 0.01, init_bias: float = -2.0) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.delta_cap = float(delta_cap)
        self.gate_margin = float(gate_margin)
        self.mlp = nn.Sequential(nn.Linear(action_dim * 3 + action_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, action_dim))
        self.init_bias = nn.Parameter(torch.full((action_dim,), float(init_bias)))

    def forward(self, z_fate: torch.Tensor, z_eva: torch.Tensor, evidence_confidence: torch.Tensor, y_action: torch.Tensor | None = None, train_mode: bool = True) -> dict[str, torch.Tensor]:
        uncertainty = 1.0 - (torch.sigmoid(z_fate) - 0.5).abs() * 2.0
        raw = self.mlp(torch.cat([z_fate, z_eva, uncertainty, evidence_confidence], dim=-1))
        visual_gate = torch.sigmoid(raw + self.init_bias)
        bounded_delta = torch.clamp(z_eva - z_fate, -self.delta_cap, self.delta_cap)
        z_actor = z_fate + visual_gate * bounded_delta
        out = {"visual_gate": visual_gate, "z_actor": z_actor, "bounded_delta": bounded_delta, "action_uncertainty": uncertainty, "gate_target": torch.zeros_like(visual_gate)}
        if y_action is not None:
            loss_fate = F.binary_cross_entropy_with_logits(z_fate, y_action.float(), reduction="none")
            loss_eva = F.binary_cross_entropy_with_logits(z_eva, y_action.float(), reduction="none")
            gate_target = (loss_eva < (loss_fate - self.gate_margin)).float()
            out["gate_target"] = gate_target
            out["gate_loss"] = F.binary_cross_entropy(visual_gate, gate_target)
        return out
