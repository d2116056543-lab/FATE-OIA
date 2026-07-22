from __future__ import annotations

import torch
from torch import nn


class _Probe(nn.Module):
    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).squeeze(-1)


class PRECISEPCVLProbes(nn.Module):
    """Detached equal-capacity U0/U1/U2/U3 pilot-only value probes."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.u0, self.u1, self.u2, self.u3 = _Probe(dim), _Probe(dim), _Probe(dim), _Probe(dim)

    def forward(self, base_tokens: torch.Tensor, oracle_evidence: torch.Tensor, learned_evidence: torch.Tensor, learned_exchange: torch.Tensor) -> dict[str, torch.Tensor]:
        if base_tokens.shape != oracle_evidence.shape or base_tokens.shape != learned_evidence.shape or base_tokens.shape != learned_exchange.shape:
            raise ValueError("PCVL inputs must retain one detached feature per action")
        base = base_tokens.detach()
        oracle = oracle_evidence.detach()
        learned = learned_evidence.detach()
        exchange = learned_exchange.detach()
        return {"u0": self.u0(base), "u1": self.u1(base + oracle), "u2": self.u2(base + learned), "u3": self.u3(base + learned + exchange)}
