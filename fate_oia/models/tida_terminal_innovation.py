from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class _SharedTerminalPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, query_identity: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([query_identity, history], dim=-1))


def _prediction_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.layer_norm(prediction, (prediction.shape[-1],))
    tgt = F.layer_norm(target, (target.shape[-1],))
    huber = F.smooth_l1_loss(pred, tgt, reduction="none").mean(-1)
    cosine = 1.0 - F.cosine_similarity(pred, tgt, dim=-1)
    return 0.5 * huber + 0.5 * cosine


class TIDATerminalInnovation(nn.Module):
    def __init__(self, dim: int = 384, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)
        self.history_predictor = _SharedTerminalPredictor(dim)
        self.innovation_norm = nn.LayerNorm(dim)

    @property
    def no_history_predictor(self) -> nn.Module:
        return self.history_predictor

    def forward(
        self,
        query_identity: torch.Tensor,
        history_summary: torch.Tensor,
        terminal_target: torch.Tensor,
        history_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target = terminal_target.detach()
        prediction_history = self.history_predictor(query_identity, history_summary)
        prediction_no_history = self.no_history_predictor(query_identity, torch.zeros_like(history_summary))
        error_history = _prediction_distance(prediction_history, target)
        error_no_history = _prediction_distance(prediction_no_history, target)
        reliability = ((error_no_history - error_history) / (error_no_history + self.eps)).clamp(0.0, 1.0).detach()
        valid = history_valid.to(dtype=torch.bool).view(-1, 1)
        reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        innovation = reliability[..., None] * self.innovation_norm(prediction_history - prediction_no_history)
        innovation = torch.where(valid[..., None], innovation, torch.zeros_like(innovation))
        return {
            "terminal_prediction_history": prediction_history,
            "terminal_prediction_no_history": prediction_no_history,
            "terminal_error_history": error_history,
            "terminal_error_no_history": error_no_history,
            "innovation_reliability": reliability,
            "innovation_token": innovation,
            "terminal_no_history_optimized": torch.tensor(False, device=target.device),
        }
