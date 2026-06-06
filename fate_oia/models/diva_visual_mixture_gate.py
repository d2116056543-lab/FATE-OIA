from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _macro_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    pred = (torch.sigmoid(logits) >= threshold).float()
    y = targets.float()
    tp = (pred * y).sum(0)
    fp = (pred * (1.0 - y)).sum(0)
    fn = ((1.0 - pred) * y).sum(0)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-9)
    return f1.mean()


def branch_safe_guarded_action(
    z_fate: torch.Tensor,
    z_actor: torch.Tensor,
    y_action: torch.Tensor | None = None,
    tolerance: float = 0.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select actor only when it does not underperform the FATE/base branch."""
    if y_action is None:
        return z_actor, {"guarded_source": "actor_unlabeled", "actor_minus_fate_mF1": None}
    fate_mf1 = _macro_f1(z_fate.detach(), y_action.detach())
    actor_mf1 = _macro_f1(z_actor.detach(), y_action.detach())
    use_actor = bool(actor_mf1 >= fate_mf1 - float(tolerance))
    guarded = z_actor if use_actor else z_fate
    return guarded, {
        "guarded_source": "actor" if use_actor else "fate",
        "Act_fate_mF1": float(fate_mf1.cpu()),
        "Act_actor_mF1": float(actor_mf1.cpu()),
        "actor_minus_fate_mF1": float((actor_mf1 - fate_mf1).cpu()),
    }


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
