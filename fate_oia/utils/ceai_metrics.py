from __future__ import annotations

import torch

from fate_oia.engine.eval_snna25 import evaluate_snna25


def eval_action_reason(action_logits: torch.Tensor, reason_logits: torch.Tensor, labels: torch.Tensor, action_dim: int, threshold: float = 0.5) -> dict:
    logits = torch.cat([action_logits.detach().cpu(), reason_logits.detach().cpu()], dim=1)
    return evaluate_snna25(logits, labels.detach().cpu(), action_dim, threshold_mode="fixed", fixed_threshold=threshold)["metrics"]
