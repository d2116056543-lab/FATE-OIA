from __future__ import annotations

import torch
from torch import Tensor, nn

from .dice_license_predictor import DICELicensePredictor


class DICEDirectionalHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, probes_per_action: int = 4,
                 c_max_per_atom: float = 0.08, total_action_cap: float = 0.25) -> None:
        super().__init__()
        self.c_max_per_atom, self.total_action_cap = float(c_max_per_atom), float(total_action_cap)
        self.support_weight = nn.Parameter(torch.zeros(action_dim, dim))
        self.counter_weight = nn.Parameter(torch.zeros(action_dim, dim))
        self.license = DICELicensePredictor(dim)

    def forward(self, centered_token: Tensor, base_action_logits: Tensor, legacy_contribution: Tensor,
                *, evidence_map: Tensor | None = None, agreement: Tensor | None = None,
                confidence: Tensor | None = None) -> dict[str, Tensor]:
        raw_support = torch.einsum("bakd,ad->bak", centered_token, self.support_weight)
        raw_counter = torch.einsum("bakd,ad->bak", centered_token, self.counter_weight)
        support_magnitude = torch.nn.functional.softplus(raw_support)
        counter_magnitude = torch.nn.functional.softplus(raw_counter)
        baseline = torch.nn.functional.softplus(torch.zeros_like(raw_support))
        if evidence_map is None:
            shape = centered_token.shape[:-1]
            evidence_map = centered_token.new_full((*shape, 1), 1.0)
            agreement = centered_token.new_zeros(shape)
            confidence = centered_token.new_zeros(shape)
        licenses = self.license(centered_token, evidence_map, agreement, confidence,
                                base_action_logits, legacy_contribution)
        licensed = (licenses["license_support_hat"] * (support_magnitude - baseline)
                    - licenses["license_counter_hat"] * (counter_magnitude - baseline))
        atom = self.c_max_per_atom * torch.tanh(licensed / self.c_max_per_atom)
        delta = self.total_action_cap * torch.tanh(atom.sum(-1) / self.total_action_cap)
        return {**licenses, "support_magnitude": support_magnitude, "counter_magnitude": counter_magnitude,
                "atom_correction": atom, "dice_action_delta": delta,
                "dice_action_logits": base_action_logits + delta}
