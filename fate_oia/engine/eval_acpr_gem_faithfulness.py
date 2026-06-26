from __future__ import annotations

import torch


def deletion_drop(original_logits: torch.Tensor, deleted_logits: torch.Tensor) -> float:
    return float((torch.sigmoid(original_logits) - torch.sigmoid(deleted_logits)).abs().mean().detach().cpu())
