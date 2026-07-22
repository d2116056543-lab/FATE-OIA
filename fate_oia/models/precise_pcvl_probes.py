from __future__ import annotations

import torch
from torch import nn


class _Probe(nn.Module):
    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class PRECISEPCVLProbes(nn.Module):
    """Detached equal-capacity U0/U1/U2/U3 pilot-only value probes."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.u0, self.u1, self.u2, self.u3 = _Probe(dim), _Probe(dim), _Probe(dim), _Probe(dim)

    def forward(self, base_tokens: torch.Tensor, oracle_evidence: torch.Tensor, learned_evidence: torch.Tensor, learned_exchange: torch.Tensor) -> dict[str, torch.Tensor]:
        base = base_tokens.detach().mean(dim=1)
        oracle = oracle_evidence.detach().mean(dim=1)
        learned = learned_evidence.detach().mean(dim=1)
        exchange = learned_exchange.detach().mean(dim=1)
        return {"u0": self.u0(base), "u1": self.u1(base + oracle), "u2": self.u2(base + learned), "u3": self.u3(base + learned + exchange)}
