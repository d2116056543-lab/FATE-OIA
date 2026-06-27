from __future__ import annotations

import torch


def influence_delta(original_logits: torch.Tensor, intervened_logits: torch.Tensor) -> torch.Tensor:
    return original_logits.softmax(-1).max(-1).values - intervened_logits.softmax(-1).max(-1).values

