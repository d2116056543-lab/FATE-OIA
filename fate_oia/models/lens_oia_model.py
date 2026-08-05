from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from .lens_action_reread import LENSActionReread
from .lens_adaptive_evidence import LENSAdaptiveEvidence
from .lens_annotation_emission import LENSAnnotationEmission
from .lens_calalign_foundation import LENSCalAlignFoundation
from .lens_latent_state import LENSLatentState


class LENSOIAModel(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, use_mock_dino: bool = False, **foundation_kwargs: Any) -> None:
        super().__init__()
        self.foundation = LENSCalAlignFoundation(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_mock_dino=use_mock_dino, **foundation_kwargs)
        self.adaptive_evidence = LENSAdaptiveEvidence(dim=dim, reason_dim=reason_dim)
        self.latent_state = LENSLatentState(dim=dim, reason_dim=reason_dim)
        self.annotation_emission = LENSAnnotationEmission(reason_dim=reason_dim)
        self.action_reread = LENSActionReread(dim=dim, action_dim=action_dim, reason_dim=reason_dim)

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.foundation.encode_images(images)

    def decode_from_field(self, field: dict[str, Any], *, progress: float, mechanism_ablation: str = "none", return_state_variants: bool = True) -> dict[str, Any]:
        source = self.foundation.decode_field(field)
        evidence = self.adaptive_evidence(source["reason_nodes_source"], source["patch_tokens_by_layer"])
        if mechanism_ablation == "evidence_map_shuffle":
            evidence["evidence_map"] = evidence["evidence_map"].flip(-1)
        latent = self.latent_state(
            source["reason_nodes_source"], evidence["evidence_token"], source["reason_visual_source"],
            evidence["evidence_null_mass"], evidence["evidence_entropy"], evidence["evidence_snr"], progress=progress,
        )
        emission = self.annotation_emission(latent["state_prob"], source["reason_logits_source"], progress=progress)
        action_reread = self.action_reread(
            action_nodes=source["action_nodes_source"], detail_tokens=source["patch_tokens_by_layer"][:, -1],
            source_action_attention=source["label_attention_source"][:, :4], evidence_map=evidence["evidence_map"],
            evidence_token=evidence["evidence_token"], state_prob=latent["state_prob"], state_token=latent["state_token"],
            action_logits_base=source["action_logits_source"], progress=progress,
        )
        if mechanism_ablation in {"reread_off", "latent_state_off", "emission_identity"}:
            action_reread["action_logits_final"] = source["action_logits_source"]
        out = {**source, **evidence, **latent, **emission, **action_reread}
        out["action_logits_base"] = source["action_logits_source"]
        out["reason_prob_formal"] = out["reason_logits_formal"].sigmoid()
        if not return_state_variants:
            out.pop("action_logits_state_substitution")
        return out

    def forward(self, images: Tensor, *, progress: float = 1.0) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), progress=progress)
