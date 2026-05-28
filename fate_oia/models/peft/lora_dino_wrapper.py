from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float | None = None) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.rank = int(rank)
        self.alpha = float(alpha if alpha is not None else rank)
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = torch.nn.functional.linear(torch.nn.functional.linear(x, self.lora_a), self.lora_b) * (self.alpha / max(1, self.rank))
        return self.base(x) + update


def _wrap_linears(module: nn.Module, names: tuple[str, ...], rank: int, alpha: float | None) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in names:
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += _wrap_linears(child, names, rank, alpha)
    return count


def apply_lora_to_last_blocks(backbone: nn.Module, last_n: int = 4, rank: int = 4, alpha: float | None = None) -> int:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("backbone has no blocks attribute")
    count = 0
    for block in list(blocks)[-int(last_n):]:
        count += _wrap_linears(block, ("qkv", "proj"), rank, alpha)
    return count
