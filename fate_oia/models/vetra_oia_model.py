from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .vetra_visual_factor_transport import VETRAVisualFactorTransport


class VETRAOIAModel(nn.Module):
    def __init__(self, base_model: nn.Module, predicate_names: list[str], grammar_path: str,
                 dim: int = 384, num_layers: int = 3, correction_cap: float = .20,
                 null_route_prior: float = .50,
                 base_forward_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        self.base_forward_kwargs = base_forward_kwargs or {"action_scale": 1.0, "reason_scale": .60}
        self.transport = VETRAVisualFactorTransport(
            predicate_names, grammar_path, dim=dim, num_layers=num_layers,
            correction_cap=correction_cap, null_route_prior=null_route_prior)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

    def decode_base_output(self, base: dict[str, Any], *, alpha: float,
                           semantic_shuffle: bool = False, visual_shuffle: bool = False,
                           force_null_only: bool = False, named_factors_off: bool = False,
                           unnamed_factors_off: bool = False, support_route_off: bool = False,
                           counter_route_off: bool = False, predicate_off: bool = False,
                           reliability_off: bool = False) -> dict[str, Any]:
        routed = self.transport(
            base["patch_tokens_by_layer_raw"], base["action_nodes_primary"], base["reason_nodes_primary"],
            base["predicate_tokens"], base["predicate_attention"], base["predicate_probs"],
            base.get("predicate_layer_weights"), alpha=alpha, semantic_shuffle=semantic_shuffle,
            visual_shuffle=visual_shuffle, force_null_only=force_null_only,
            named_factors_off=named_factors_off, unnamed_factors_off=unnamed_factors_off,
            support_route_off=support_route_off, counter_route_off=counter_route_off,
            predicate_off=predicate_off, reliability_off=reliability_off)
        action_base = base["action_logits_final"].detach()
        reason_base = base["reason_logits_final"].detach()
        return {**base, **routed,
                "action_logits_base": action_base,
                "action_logits_final": action_base + routed["vetra_action_delta"],
                "reason_logits_base": reason_base,
                "reason_logits_final": reason_base,
                "reason_identity_max_abs": torch.zeros((), device=reason_base.device, dtype=reason_base.dtype),
                "vetra_alpha": action_base.new_tensor(float(alpha))}

    def forward(self, images: Tensor, *, alpha: float = 1.0, **ablation) -> dict[str, Any]:
        with torch.no_grad():
            base = self.base_model(images, **self.base_forward_kwargs)
        return self.decode_base_output(base, alpha=alpha, **ablation)
