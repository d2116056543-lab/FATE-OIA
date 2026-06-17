from __future__ import annotations

import copy
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.995) -> None:
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if k in msd and torch.is_floating_point(v):
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            elif k in msd:
                v.copy_(msd[k])
