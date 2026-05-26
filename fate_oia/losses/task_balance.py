from __future__ import annotations

import torch
from torch import nn


class UncertaintyTaskBalancer(nn.Module):
    """Kendall-style uncertainty weighting for a small fixed set of task losses."""

    def __init__(self, task_names: list[str] | tuple[str, ...]) -> None:
        super().__init__()
        if not task_names:
            raise ValueError("task_names must not be empty")
        self.task_names = tuple(str(x) for x in task_names)
        self.log_vars = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in self.task_names})

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total: torch.Tensor | None = None
        components: dict[str, torch.Tensor] = {}
        for name in self.task_names:
            if name not in losses:
                continue
            loss = losses[name]
            log_var = self.log_vars[name]
            weighted = torch.exp(-log_var) * loss + log_var
            components[f"task_balance_{name}_weighted"] = weighted
            components[f"task_balance_{name}_log_var"] = log_var
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No matching losses were provided to UncertaintyTaskBalancer.")
        return total, components


__all__ = ["UncertaintyTaskBalancer"]
