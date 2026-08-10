from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .dice_atom_reconstructor import DICEAtomReconstructor
from .dice_directional_head import DICEDirectionalHead


class DICEOIAModel(nn.Module):
    def __init__(self, base_model: nn.Module, dim: int = 384, num_layers: int = 3,
                 num_predicates: int = 32, probes_per_action: int = 4,
                 predicate_strength_max: float = .20, predicate_presence_floor: float = .30,
                 c_max_per_atom: float = .08, total_action_cap: float = .25,
                 base_forward_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.base_forward_kwargs = base_forward_kwargs or {
            "semantic_share_license": 0.0, "action_scale": 0.0,
            "reason_budget": 0.0, "compatibility_mode": True,
        }
        self.atom_reconstructor = DICEAtomReconstructor(
            dim, num_layers, num_predicates, predicate_strength_max=predicate_strength_max,
            predicate_presence_floor=predicate_presence_floor)
        self.directional_head = DICEDirectionalHead(dim, 4, probes_per_action, c_max_per_atom, total_action_cap)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def decode_base_output(self, base: dict[str, Any]) -> dict[str, Any]:
        atoms = self.atom_reconstructor(
            base["evidence_token"], base["conditioned_patch_layers"], base["predicate_attention"],
            base["predicate_probs"], base["ego_region_masks"])
        direction = self.directional_head(
            atoms["centered_token"], base["action_logits_final"].detach(),
            base["bounded_contribution"].detach(), evidence_map=atoms["coherent_map"],
            agreement=atoms["predicate_agreement"], confidence=atoms["predicate_confidence"])
        reason = base["reason_logits_final"].detach()
        return {**base, **atoms, **direction,
                "action_logits_base": base["action_logits_final"].detach(),
                "action_logits_final": direction["dice_action_logits"],
                "reason_logits_base": reason, "reason_logits_final": reason,
                "reason_identity_max_abs": torch.zeros((), device=reason.device, dtype=reason.dtype)}

    def rerun_dice_from_conditioned(self, base: dict[str, Any], conditioned_patch_layers: Tensor) -> dict[str, Any]:
        atoms = self.atom_reconstructor(
            base["evidence_token"], conditioned_patch_layers, base["predicate_attention"],
            base["predicate_probs"], base["ego_region_masks"])
        direction = self.directional_head(
            atoms["centered_token"], base["action_logits_final"].detach(),
            base["bounded_contribution"].detach(), evidence_map=atoms["coherent_map"],
            agreement=atoms["predicate_agreement"], confidence=atoms["predicate_confidence"])
        return {**atoms, **direction, "dino_calls_cf_event": 0}

    def forward(self, images: Tensor) -> dict[str, Any]:
        with torch.no_grad():
            base = self.base_model(images, **self.base_forward_kwargs)
        return self.decode_base_output(base)
