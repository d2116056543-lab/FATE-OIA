from __future__ import annotations

import torch
import torch.nn.functional as F


class ActionParetoGate:
    """Train-calib-only gate updater for action utility deltas."""

    def __init__(self, action_dim: int = 4, gate_ema: float = 0.20, action_margin: float = 0.002, min_support: int = 5) -> None:
        self.action_dim = int(action_dim)
        self.gate_ema = float(gate_ema)
        self.action_margin = float(action_margin)
        self.min_support = int(min_support)
        self.r2a_gate = torch.zeros(self.action_dim)
        self.pred_gate = torch.zeros(self.action_dim)
        self.last_stats: dict = {}

    @staticmethod
    def _label_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none").mean(0)

    def update(self, fallback_logits: torch.Tensor, r2a_candidate_logits: torch.Tensor, pred_candidate_logits: torch.Tensor, targets: torch.Tensor) -> dict:
        fb = self._label_loss(fallback_logits, targets)
        r2a = self._label_loss(r2a_candidate_logits, targets)
        pred = self._label_loss(pred_candidate_logits, targets)
        support = targets.float().sum(0).cpu()
        r2a_help = (fb - r2a).detach().cpu()
        pred_help = (fb - pred).detach().cpu()
        r2a_open = ((r2a_help > self.action_margin) & (support >= self.min_support)).float()
        pred_open = ((pred_help > self.action_margin) & (support >= self.min_support)).float()
        self.r2a_gate = (1.0 - self.gate_ema) * self.r2a_gate + self.gate_ema * r2a_open
        self.pred_gate = (1.0 - self.gate_ema) * self.pred_gate + self.gate_ema * pred_open
        self.last_stats = {
            "support": support.tolist(),
            "r2a_loss_improvement": r2a_help.tolist(),
            "pred_loss_improvement": pred_help.tolist(),
            "r2a_gate": self.r2a_gate.tolist(),
            "pred_gate": self.pred_gate.tolist(),
            "utility_helped_action_labels": [int(i) for i, x in enumerate((r2a_open + pred_open) > 0)],
            "utility_hurt_action_labels": [int(i) for i, x in enumerate((r2a_help < -self.action_margin) | (pred_help < -self.action_margin)) if bool(x)],
            "source": "train_calib_only",
        }
        return self.last_stats

    def state_dict(self) -> dict:
        return {"r2a_gate": self.r2a_gate.clone(), "pred_gate": self.pred_gate.clone(), "last_stats": self.last_stats}

    def load_state_dict(self, state: dict) -> None:
        if "r2a_gate" in state: self.r2a_gate = state["r2a_gate"].clone().cpu()
        if "pred_gate" in state: self.pred_gate = state["pred_gate"].clone().cpu()
        self.last_stats = dict(state.get("last_stats", {}))
