from __future__ import annotations

from typing import TypedDict

import torch
from torch import nn


class HeadOutput(TypedDict, total=False):
    logits: torch.Tensor
    action_logits: torch.Tensor
    reason_logits: torch.Tensor
    label_tokens: torch.Tensor | None
    attention: torch.Tensor | None
    aux_losses: dict[str, torch.Tensor]
    raw_logits: torch.Tensor
    raw_action_logits: torch.Tensor
    raw_reason_logits: torch.Tensor


class BaseOIAHead(nn.Module):
    action_dim: int
    reason_dim: int

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs) -> HeadOutput:
        raise NotImplementedError


def split_logits(logits: torch.Tensor, action_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, :action_dim], logits[:, action_dim:]


def with_common_fields(
    logits: torch.Tensor,
    action_dim: int,
    *,
    label_tokens: torch.Tensor | None = None,
    attention: torch.Tensor | None = None,
    aux_losses: dict[str, torch.Tensor] | None = None,
    **extra,
) -> HeadOutput:
    action_logits, reason_logits = split_logits(logits, action_dim)
    out: HeadOutput = {
        "logits": logits,
        "action_logits": action_logits,
        "reason_logits": reason_logits,
        "label_tokens": label_tokens,
        "attention": attention,
        "aux_losses": aux_losses or {},
    }
    out.update(extra)
    return out
