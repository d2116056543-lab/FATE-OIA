from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .lens_action_reread import LENSActionReread
from .lens_adaptive_evidence import LENSAdaptiveEvidence
from .lens_annotation_emission import LENSAnnotationEmission
from .lens_calalign_foundation import LENSCalAlignFoundation
from .lens_latent_state import LENSLatentState


class LENSOIAModel(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, use_mock_dino: bool = False, factor_chunk_size: int = 21, evidence_tau_min: float = 0.35, evidence_tau_max: float = 2.0, evidence_topk: int = 32, region_bias_abs_max: float = 2.0, action_logit_cap: float = 20.0, **foundation_kwargs: Any) -> None:
        super().__init__()
        self.foundation = LENSCalAlignFoundation(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_mock_dino=use_mock_dino, **foundation_kwargs)
        self.factor_chunk_size=factor_chunk_size
        self.adaptive_evidence = LENSAdaptiveEvidence(dim=dim, reason_dim=reason_dim,tau_min=evidence_tau_min,tau_max=evidence_tau_max,topk=evidence_topk,region_bias_abs_max=region_bias_abs_max)
        self.latent_state = LENSLatentState(dim=dim, reason_dim=reason_dim)
        self.annotation_emission = LENSAnnotationEmission(reason_dim=reason_dim)
        self.action_reread = LENSActionReread(dim=dim, action_dim=action_dim, reason_dim=reason_dim,cap=action_logit_cap)

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.foundation.encode_images(images)

    def decode_from_field(self, field: dict[str, Any], *, progress: float, mechanism_ablation: str = "none", return_state_variants: bool = True) -> dict[str, Any]:
        source = self.foundation.decode_field(field)
        evidence = self.adaptive_evidence(source["reason_nodes_source"], source["patch_tokens_by_layer"])
        if mechanism_ablation == "evidence_map_shuffle":
            evidence["evidence_map"] = evidence["evidence_map"].flip(-1)
        if mechanism_ablation == "wrong_factor":
            for key in ("evidence_map","evidence_null_mass","evidence_token","evidence_temperature","evidence_snr","evidence_entropy","evidence_score_mean","evidence_score_std","evidence_topk_gap"):
                evidence[key]=evidence[key].roll(1,dims=1)
        latent_progress=0.0 if mechanism_ablation == "latent_state_off" else progress
        latent = self.latent_state(
            source["reason_nodes_source"], evidence["evidence_token"], source["reason_visual_source"],
            evidence["evidence_null_mass"], evidence["evidence_entropy"], evidence["evidence_snr"], progress=latent_progress,
        )
        if mechanism_ablation == "unknown_off":
            known=latent["state_prob"][...,:2]; known=known/known.sum(-1,keepdim=True).clamp_min(1e-8)
            latent["state_prob"]=torch.cat([known,torch.zeros_like(known[...,:1])],-1)
            latent["state_positive_prob"],latent["state_counter_prob"]=known[...,0],known[...,1]
            latent["state_unknown_prob"]=torch.zeros_like(known[...,0]); latent["state_observability"]=torch.ones_like(known[...,0])
            latent["state_token"]=evidence["evidence_token"]+torch.einsum("brs,rsd->brd",latent["state_prob"],latent["state_embeddings"])
        emission_progress=0.0 if mechanism_ablation == "emission_identity" else progress
        emission = self.annotation_emission(latent["state_prob"], source["reason_logits_source"], progress=emission_progress)
        # The action base consumes observable latent log-odds, never the annotation-emission branch.
        clean_log_odds = (1.0 - latent["state_unknown_prob"]) * torch.log(
            latent["state_positive_prob"].clamp_min(1e-8) / latent["state_counter_prob"].clamp_min(1e-8)
        )
        action_reason_latent = self.foundation.trunk.reason_to_action(clean_log_odds)
        action_logits_base = source["action_logits_source"] if float(progress) == 0.0 else (
            source["action_fusion_gate_source"] * source["action_visual_source"]
            + (1.0 - source["action_fusion_gate_source"]) * action_reason_latent
        )
        action_reread = self.action_reread(
            action_nodes=source["action_nodes_source"], detail_tokens=source["patch_tokens_by_layer"][:, -1],
            source_action_attention=source["label_attention_source"][:, :4], evidence_map=evidence["evidence_map"],
            evidence_token=evidence["evidence_token"], state_prob=latent["state_prob"], state_token=latent["state_token"],
            state_embeddings=latent["state_embeddings"],
            action_logits_base=action_logits_base, progress=progress,
            factor_chunk_size=self.factor_chunk_size,
        )
        if mechanism_ablation in {"reread_off", "factor_off"}:
            action_reread["action_logits_final"] = action_logits_base
        if mechanism_ablation == "factor_only":
            action_reread["action_logits_final"] = action_reread["action_logits_factor_aux"]
        out = {**source, **evidence, **latent, **emission, **action_reread}
        out["clean_observable_log_odds"] = clean_log_odds
        out["action_reason_latent"] = action_reason_latent
        out["action_logits_base"] = action_logits_base
        out["reason_prob_formal"] = out["reason_logits_formal"].sigmoid()
        if not return_state_variants:
            out.pop("action_logits_state_substitution")
        return out

    def forward(self, images: Tensor, *, progress: float = 1.0) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), progress=progress)

    def decode_branches_from_field(self, field: dict[str, Any], *, progress: float, base_output: dict[str, Any] | None = None) -> dict[str, dict[str, Tensor]]:
        base=base_output if base_output is not None else self.decode_from_field(field,progress=progress)
        branches={
            "source_calalign":{"action":base["action_logits_source"],"reason":base["reason_logits_source"]},
            "lens_base":{"action":base["action_logits_base"],"reason":base["reason_logits_latent"]},
            "lens_final":{"action":base["action_logits_final"],"reason":base["reason_logits_formal"]},
            "factor_only":{"action":base["action_logits_factor_aux"],"reason":base["reason_logits_formal"]},
        }
        for name in ("reread_off","latent_state_off","unknown_off","emission_identity","evidence_map_shuffle","wrong_factor"):
            value=self.decode_from_field(field,progress=progress,mechanism_ablation=name,return_state_variants=False)
            branches[name]={"action":value["action_logits_final"],"reason":value["reason_logits_formal"]}
        return branches
