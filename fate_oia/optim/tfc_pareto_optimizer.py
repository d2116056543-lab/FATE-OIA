from __future__ import annotations

import torch


class TFCParetoOptimizer:
    """Action-priority PCGrad helper.

    The training code may use structural firewall instead of full PCGrad for
    speed, but this class is functional and logs projection evidence.
    """

    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = [p for p in params if p.requires_grad]

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)

    def flatten_grads(self) -> torch.Tensor:
        chunks = []
        for p in self.params:
            if p.grad is not None:
                chunks.append(p.grad.detach().flatten())
        if not chunks:
            return torch.zeros(1)
        return torch.cat(chunks)
