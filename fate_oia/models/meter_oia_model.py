from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .meter_calalign_foundation import METERCalAlignFoundation
from .meter_reason_decoder import METERPrivateReasonDecoder
from .meter_semantic_action import METERSemanticActionPeer
from .meter_signed_factors import METERsignedFactors


class METEROIAModel(nn.Module):
    """Formal RGB-only METER graph with an action firewall around private reasons."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        factor_rank: int = 16,
    ) -> None:
        super().__init__()
        self.foundation = METERCalAlignFoundation(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
        )
        self.signed_factors = METERsignedFactors(dim=dim, factor_dim=reason_dim, num_layers=len(selected_layers), rank=factor_rank)
        self.action_peer = METERSemanticActionPeer(dim=dim, action_dim=action_dim, factor_dim=reason_dim)
        self.reason_decoder = METERPrivateReasonDecoder(dim=dim, reason_dim=reason_dim, action_dim=action_dim)
        self.reason_decoder.initialize_from_foundation(self.foundation)
        self.register_buffer("meta_share_weight", Tensor().new_zeros(reason_dim), persistent=True)

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.foundation.encode_images(images)

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        progress: float = 1.0,
        diagnostic_modes: tuple[str, ...] = (),
        factor_parameter_override: dict[str, Tensor] | None = None,
        meta_share_weight_override: Tensor | None = None,
    ) -> dict[str, Any]:
        base = self.foundation.decode_foundation(field)
        factors = self.signed_factors(
            base["factor_base_nodes"],
            base["patch_tokens_by_layer"],
            progress=progress,
            meta_share_weight=self.meta_share_weight,
            factor_parameter_override=factor_parameter_override,
        )
        factor_action_tokens = factors["factor_action_tokens"]
        factor_reliability = factors["factor_reliability"]
        factor_to_reason_tokens = factors["factor_to_reason_tokens"]
        support_map = factors["factor_support_map"]
        counter_map = factors["factor_counter_map"]
        if "factor_off" in diagnostic_modes:
            factor_action_tokens = torch.zeros_like(factor_action_tokens)
            factor_reliability = torch.zeros_like(factor_reliability)
            factor_to_reason_tokens = torch.zeros_like(factor_to_reason_tokens)
            support_map = torch.zeros_like(support_map)
            counter_map = torch.zeros_like(counter_map)
        elif "factor_shuffle" in diagnostic_modes:
            # Shuffle factor identity within each image, not the batch.  A
            # batch roll would compare different images and make this
            # diagnostic confound evidence corruption with sample leakage.
            factor_action_tokens = torch.roll(factor_action_tokens, shifts=1, dims=1)
            factor_reliability = torch.roll(factor_reliability, shifts=1, dims=1)
            factor_to_reason_tokens = torch.roll(factor_to_reason_tokens, shifts=1, dims=1)
            support_map = torch.roll(support_map, shifts=1, dims=1)
            counter_map = torch.roll(counter_map, shifts=1, dims=1)
        elif "support_only" in diagnostic_modes:
            factor_action_tokens = factors["factor_support_detail"]
            factor_reliability = factors["factor_reliability"]
        elif "counter_only" in diagnostic_modes:
            factor_action_tokens = factors["factor_counter_detail"]
            factor_reliability = factors["factor_reliability"]
        elif "meta_off" in diagnostic_modes:
            factor_action_tokens = factors["factor_core_tokens"]
            factor_to_reason_tokens = factors["factor_core_tokens"]
        action = self.action_peer(
            base["action_logits_calalign"],
            base["action_nodes"],
            factor_action_tokens,
            factor_reliability,
            progress=progress,
        )
        reason_action_logits = action["action_logits_final"]
        if "decision_context_off" in diagnostic_modes:
            reason_action_logits = torch.zeros_like(reason_action_logits)
        # Rebuild the private input at the model boundary.  The detached
        # action-factor view is the ordinary reason input; only the explicit
        # meta bridge may carry a reason gradient into a meta adapter.
        share_weight = self.meta_share_weight if meta_share_weight_override is None else meta_share_weight_override
        omega = share_weight.to(device=factor_to_reason_tokens.device, dtype=factor_to_reason_tokens.dtype).view(1, -1, 1).clamp(0.0, 1.0)
        if "meta_off" in diagnostic_modes:
            reason_factor_tokens = factors["factor_core_tokens"].detach()
        else:
            reason_factor_tokens = factors["factor_action_tokens"].detach() + float(min(max(progress / 0.10, 0.0), 1.0)) * omega * (factors["factor_meta_delta"] - factors["factor_meta_delta"].detach())
        reason_support_map = support_map
        reason_counter_map = counter_map
        if "factor_context_off" in diagnostic_modes:
            reason_factor_tokens = torch.zeros_like(reason_factor_tokens)
        if "map_shuffle" in diagnostic_modes:
            reason_support_map = torch.roll(reason_support_map, shifts=1, dims=-1)
            reason_counter_map = torch.roll(reason_counter_map, shifts=1, dims=-1)
        reason = self.reason_decoder(
            patch_tokens_by_layer=base["patch_tokens_by_layer"].detach(),
            reason_logits_calalign=base["reason_logits_calalign"].detach(),
            action_logits_final=reason_action_logits,
            action_nodes=base["action_nodes"].detach(),
            factor_to_reason_tokens=reason_factor_tokens,
            factor_support_map=reason_support_map.detach(),
            factor_counter_map=reason_counter_map.detach(),
            factor_reliability=factor_reliability.detach(),
            factor_support_null=factors["factor_support_null"].detach(),
            progress=progress,
        )
        if "annotation_off" in diagnostic_modes:
            reason["reason_logits_final"] = base["reason_logits_calalign"] + self.reason_decoder._ramp(progress) * (reason["reason_logits_mix"] - base["reason_logits_calalign"])
        return {
            **field,
            **base,
            **factors,
            **action,
            **reason,
            "branch_logits": {
                "action_visual": action["action_logits_visual"],
                "action_semantic": action["action_logits_semantic"],
                "action_peer": action["action_logits_peer"],
                "action_final": action["action_logits_final"],
                "reason_calalign": base["reason_logits_calalign"],
                "reason_global": reason["reason_logits_global"],
                "reason_local": reason["reason_logits_local"],
                "reason_mix": reason["reason_logits_mix"],
                "reason_final": reason["reason_logits_final"],
            },
            "diagnostic_modes": diagnostic_modes,
        }

    def forward(
        self,
        images: Tensor,
        *,
        progress: float = 1.0,
        diagnostic_modes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), progress=progress, diagnostic_modes=diagnostic_modes)
