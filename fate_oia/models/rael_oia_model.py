"""Direct-image RAEL-OIA integration model.

This module is the sole integration owner.  It composes the already-audited
RAEL components without accepting labels, geometry annotations, text, cache
paths, or training state in the formal forward path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.func import functional_call

from fate_oia.losses.rael_grounding_losses import (
    RAELSlotAttributeHeads,
    reliable_absence_evidence,
)
from fate_oia.losses.rael_pu_losses import (
    build_pu_soft_targets,
    reason_confidence_weights,
)
from fate_oia.models.rael_action_reason_bridge import RAELActionReasonBridge
from fate_oia.models.rael_category_foundation import RAELActionCategoryFoundation
from fate_oia.models.rael_dino_field import RAELDinoFieldExtractor
from fate_oia.models.rael_multilayer_field import RAELMultiLayerField
from fate_oia.models.rael_reason_private import RAELReasonPrivateAdapter
from fate_oia.models.rael_relation_contributions import (
    RAELPairwiseContribution,
    RAELUnaryContribution,
)
from fate_oia.models.rael_semantic_reason import RAELSemanticReason
from fate_oia.models.rael_slot_ledger import RAELSlotLedger


DIM = 384
NUM_ACTIONS = 4
NUM_REASONS = 21
NUM_PUBLIC_SLOTS = 20
NUM_ENTITY_SLOTS = 12
NUM_ROAD_SLOTS = 5
NUM_LATENT_SLOTS = 3

BRANCH_NAMES = (
    "global_only",
    "global_plus_semantic_bridge",
    "unary_only",
    "pairwise_only",
    "full",
    "no_semantic_reason",
    "semantic_reason_shuffled",
    "reason_private_shuffled",
    "named_slots_only",
    "latent_slots_only",
    "global_context_only",
    "evidence_shuffled",
    "pairwise_off",
    "pu_off",
)


@dataclass(frozen=True)
class RAELVisualBundle:
    """One-DINO-call visual bundle shared by every downstream branch."""

    prepared_field: Mapping[str, Any]
    dino_outputs: Mapping[str, Any]


class _ForwardCallCounter:
    """Scoped forward-hook counter with no persistent model instrumentation."""

    def __init__(self, modules: Mapping[str, nn.Module]) -> None:
        self._modules = dict(modules)
        self._counts = {name: 0 for name in self._modules}
        self._handles: list[Any] = []

    def __enter__(self) -> _ForwardCallCounter:
        for name, module in self._modules.items():
            self._handles.append(
                module.register_forward_hook(
                    lambda _module, _inputs, _output, name=name: self._increment(name)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _increment(self, name: str) -> None:
        self._counts[name] += 1

    def record(self, name: str) -> None:
        if name not in self._counts:
            raise KeyError(f"unregistered call counter name: {name}")
        self._increment(name)

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)


class RAELPULabelPrivateHead(nn.Module):
    """Independent label-wise private probability head for P12 PU scoring."""

    parameter_owner = "pu_private"
    learning_rate = 3.0e-4

    def __init__(self, dim: int = DIM) -> None:
        super().__init__()
        if dim != DIM:
            raise ValueError(f"RAEL PU private head requires dim={DIM}")
        self.weight = nn.Parameter(torch.empty(NUM_REASONS, dim))
        self.bias = nn.Parameter(torch.zeros(NUM_REASONS))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, private_delta: Tensor, keep_mask: Tensor) -> Tensor:
        if private_delta.ndim != 3 or private_delta.shape[1:] != (NUM_REASONS, DIM):
            raise ValueError("private_delta must be [B,21,384]")
        if keep_mask.shape != (1, 1, DIM):
            raise ValueError("keep_mask must be [1,1,384]")
        if private_delta.device != self.weight.device or keep_mask.device != private_delta.device:
            raise ValueError("PU private inputs must share the head device")
        keep_probability = keep_mask.float().mean().clamp_min(1.0e-6)
        dropped_private = private_delta.detach() * keep_mask / keep_probability
        return torch.einsum("brd,rd->br", dropped_private, self.weight) + self.bias


def _default_reason_schema() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "rael_reason_semantics.yaml"


def _sector_probabilities_from_masks(masks: Tensor, eps: float = 1.0e-6) -> tuple[Tensor, Tensor]:
    if masks.ndim != 4 or masks.shape[1] != NUM_PUBLIC_SLOTS:
        raise ValueError("public slot masks must be [B,20,H,W]")
    _, _, height, width = masks.shape
    x_bins = torch.arange(width, device=masks.device) * 3 // width
    y_bins = 2 - (torch.arange(height, device=masks.device) * 3 // height)
    horizontal_mass = torch.stack(
        [masks[..., x_bins == index].sum(dim=(-1, -2)) for index in range(3)],
        dim=-1,
    )
    depth_mass = torch.stack(
        [masks[..., y_bins == index, :].sum(dim=(-1, -2)) for index in range(3)],
        dim=-1,
    )
    horizontal = horizontal_mass / horizontal_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    depth = depth_mass / depth_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    uniform = torch.full_like(horizontal, 1.0 / 3.0)
    horizontal = torch.where(
        horizontal_mass.sum(dim=-1, keepdim=True) > eps, horizontal, uniform
    )
    depth = torch.where(depth_mass.sum(dim=-1, keepdim=True) > eps, depth, uniform)
    return horizontal, depth


def _target_partition_ratios(
    unary: Tensor,
    pairwise: Tensor,
    pair_indices: Tensor,
) -> tuple[Tensor, Tensor]:
    named_unary = unary[..., : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS].abs().sum(dim=-1)
    latent_unary = unary[..., NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS :].abs().sum(dim=-1)
    pair_left, pair_right = pair_indices.unbind(dim=-1)
    named_pair = (
        (pair_left < NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS)
        & (pair_right < NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS)
    )
    latent_pair = ~named_pair
    named = named_unary + pairwise[..., named_pair].abs().sum(dim=-1)
    latent = latent_unary + pairwise[..., latent_pair].abs().sum(dim=-1)
    denominator = (named + latent).clamp_min(1.0e-8)
    return named / denominator, latent / denominator


def _signed_partition(
    global_logits: Tensor,
    unary: Tensor,
    pairwise: Tensor,
) -> tuple[Tensor, Tensor]:
    components = torch.cat(
        (global_logits.unsqueeze(-1), unary, pairwise),
        dim=-1,
    )
    return components.clamp_min(0.0).sum(-1), components.clamp_max(0.0).sum(-1)


class RAELOIAModel(nn.Module):
    """Full RAEL representation with explicit action/reason firewalls."""

    def __init__(
        self,
        *,
        dino_extractor: nn.Module | None = None,
        reason_schema_path: str | Path | None = None,
        dim: int = DIM,
        num_heads: int = 6,
    ) -> None:
        super().__init__()
        if dim != DIM:
            raise ValueError(f"RAEL-OIA V1 requires dim={DIM}")
        self.dim = int(dim)
        self.dino_extractor = dino_extractor or RAELDinoFieldExtractor()
        self.multilayer_field = RAELMultiLayerField(dim=self.dim)
        self.slot_ledger = RAELSlotLedger(dim=self.dim)
        self.slot_attribute_heads = RAELSlotAttributeHeads(dim=self.dim)
        self.semantic_reason = RAELSemanticReason(
            reason_schema_path or _default_reason_schema(), dim=self.dim
        )
        self.action_category = RAELActionCategoryFoundation(
            dim=self.dim, num_heads=num_heads
        )
        self.action_reason_bridge = RAELActionReasonBridge(
            dim=self.dim, num_heads=num_heads
        )
        self.reason_private = RAELReasonPrivateAdapter(dim=self.dim)
        self.pu_private_head = RAELPULabelPrivateHead(dim=self.dim)
        self.action_unary = RAELUnaryContribution(
            num_targets=NUM_ACTIONS, dim=self.dim
        )
        self.reason_unary = RAELUnaryContribution(
            num_targets=NUM_REASONS, dim=self.dim
        )
        self.action_pairwise = RAELPairwiseContribution(
            num_targets=NUM_ACTIONS, dim=self.dim
        )
        self.reason_pairwise = RAELPairwiseContribution(
            num_targets=NUM_REASONS, dim=self.dim
        )
        self.register_buffer(
            "_pu_active_labels",
            torch.zeros(NUM_REASONS, dtype=torch.bool),
            persistent=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(20260725)
        first_mask = (torch.rand(DIM, generator=generator) >= 0.15).float()
        second_mask = (torch.rand(DIM, generator=generator) >= 0.15).float()
        if torch.equal(first_mask, second_mask):
            second_mask = second_mask.roll(1)
        self.register_buffer(
            "_pu_feature_keep_view_one", first_mask.view(1, 1, DIM), persistent=False
        )
        self.register_buffer(
            "_pu_feature_keep_view_two", second_mask.view(1, 1, DIM), persistent=False
        )
        latent_first = (torch.rand(DIM, generator=generator) >= 0.15).float()
        latent_second = (torch.rand(DIM, generator=generator) >= 0.15).float()
        if torch.equal(latent_first, latent_second):
            latent_second = latent_second.roll(1)
        self.register_buffer(
            "_latent_feature_keep_view_one",
            latent_first.view(1, 1, DIM),
            persistent=False,
        )
        self.register_buffer(
            "_latent_feature_keep_view_two",
            latent_second.view(1, 1, DIM),
            persistent=False,
        )

    def set_pu_active_labels(self, active_labels: Tensor) -> None:
        if active_labels.dtype != torch.bool or active_labels.shape != (NUM_REASONS,):
            raise ValueError("active_labels must be bool [21]")
        self._pu_active_labels.copy_(active_labels.to(self._pu_active_labels.device))

    def _counted_modules(self) -> dict[str, nn.Module]:
        return {
            "slot_ledger": self.slot_ledger,
            "slot_attribute_heads": self.slot_attribute_heads,
            "semantic_reason": self.semantic_reason,
            "action_category": self.action_category,
            "action_reason_bridge": self.action_reason_bridge,
            "reason_private": self.reason_private,
            "pu_private_head": self.pu_private_head,
            "action_unary": self.action_unary,
            "reason_unary": self.reason_unary,
            "action_pairwise": self.action_pairwise,
            "reason_pairwise": self.reason_pairwise,
            "action_global_projector": self.action_category.global_head,
            "reason_global_head": self.reason_private.reason_global_head,
            "multilayer_field": self.multilayer_field,
        }

    def _encode_images_impl(
        self,
        images: Tensor,
        *,
        call_counter: _ForwardCallCounter | None,
    ) -> RAELVisualBundle:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("images must be [B,3,360,640]")
        if tuple(images.shape[1:]) != (3, 360, 640):
            raise ValueError("formal RAEL images must be [B,3,360,640]")
        dino_outputs = self.dino_extractor(images)
        if not isinstance(dino_outputs, Mapping):
            raise TypeError("DINO extractor must return a mapping")
        if int(dino_outputs.get("dino_call_count", 0)) != 1:
            raise RuntimeError("each RAEL encode_images call must invoke DINO exactly once")
        prepared = self.multilayer_field.precompute(
            dino_outputs["patch_tokens_by_layer"],
            dino_outputs["cls_tokens_by_layer"],
            tuple(dino_outputs["grid_hw"]),
        )
        if call_counter is not None:
            # ``precompute`` is a formal method rather than nn.Module.forward.
            # Count the successful invocation at its real callsite.
            call_counter.record("multilayer_field")
        return RAELVisualBundle(prepared_field=prepared, dino_outputs=dino_outputs)

    def encode_images(self, images: Tensor) -> RAELVisualBundle:
        return self._encode_images_impl(images, call_counter=None)

    @staticmethod
    def _validate_diagnostic_modes(diagnostic_modes: Sequence[str]) -> tuple[str, ...]:
        modes = tuple(str(mode) for mode in diagnostic_modes)
        unknown = sorted(set(modes).difference(BRANCH_NAMES))
        if unknown:
            raise ValueError(f"unknown diagnostic mode(s): {unknown}")
        return modes

    @staticmethod
    def _canonical_slot_masks(evidence_slots: Tensor, field_keys: Tensor) -> Tensor:
        """Build the sole trainable public-slot mask from the E04 boundary.

        E04-R2 stops gradients through prepared visual keys. Every formal mask
        VJP therefore enters only through ``evidence_slots`` and P13 admission;
        no trainable projection exists after the boundary. Ledger masks remain
        detached diagnostics only.
        """

        if evidence_slots.ndim != 3 or evidence_slots.shape[1:] != (
            NUM_PUBLIC_SLOTS,
            DIM,
        ):
            raise ValueError("canonical masks require evidence_slots [B,20,384]")
        if field_keys.ndim != 4 or field_keys.shape[1:] != (4, 45 * 80, DIM):
            raise ValueError("canonical masks require field_keys [B,4,3600,384]")
        if field_keys.shape[0] != evidence_slots.shape[0]:
            raise ValueError("canonical masks require matching evidence/key batches")

        # This fixed-temperature similarity has E as its only trainable input.
        keys = F.normalize(field_keys.detach().float().mean(dim=1), dim=-1)
        queries = F.normalize(evidence_slots.float(), dim=-1)
        similarity = torch.einsum("bjd,bnd->bjn", queries, keys)
        masks = torch.softmax(similarity / 0.25, dim=1)
        return masks.reshape(evidence_slots.shape[0], NUM_PUBLIC_SLOTS, 45, 80)

    def _slot_attributes(
        self,
        *,
        evidence_slots: Tensor,
        canonical_masks: Tensor,
        q_ground: Tensor | None = None,
        q_view: Tensor | None = None,
        q_view_sector: Tensor | None = None,
    ) -> dict[str, Tensor | dict[str, Tensor]]:
        """Read every formal mask-derived attribute from the E04-R2 mask."""

        slot_tokens = evidence_slots
        slot_masks = canonical_masks
        if slot_masks.shape != (
            slot_tokens.shape[0],
            NUM_PUBLIC_SLOTS,
            45,
            80,
        ):
            raise ValueError("canonical slot masks must be [B,20,45,80]")
        activity = (
            slot_masks.sum(dim=(-1, -2))
            / float(slot_masks.shape[-1] * slot_masks.shape[-2])
        ).clamp(0.0, 1.0)
        entity_tokens = slot_tokens[:, :NUM_ENTITY_SLOTS]
        road_tokens = slot_tokens[:, NUM_ENTITY_SLOTS : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS]
        entity_masks = slot_masks[:, :NUM_ENTITY_SLOTS]
        road_masks = slot_masks[:, NUM_ENTITY_SLOTS : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS]
        if q_ground is None:
            q_ground = torch.ones(
                slot_tokens.shape[:2], device=slot_tokens.device, dtype=slot_tokens.dtype
            )
        if q_view is None:
            q_view = torch.ones_like(q_ground)
        if (
            q_ground.shape != slot_tokens.shape[:2]
            or q_view.shape != slot_tokens.shape[:2]
            or q_ground.requires_grad
            or q_view.requires_grad
        ):
            raise ValueError("dynamic q_ground/q_view must be detached [B,20]")
        q_ground = q_ground.to(device=slot_tokens.device, dtype=slot_tokens.dtype)
        q_view = q_view.to(device=slot_tokens.device, dtype=slot_tokens.dtype)
        entity = self.slot_attribute_heads(
            entity_tokens,
            entity_masks,
            road_tokens,
            road_masks,
            q_ground=q_ground[:, :NUM_ENTITY_SLOTS],
            q_view=q_view[:, :NUM_ENTITY_SLOTS],
        )
        horizontal, depth = _sector_probabilities_from_masks(slot_masks)
        horizontal = horizontal.clone()
        depth = depth.clone()
        horizontal[:, :NUM_ENTITY_SLOTS] = entity["horizontal_sector_probs"]
        depth[:, :NUM_ENTITY_SLOTS] = entity["depth_sector_probs"]

        rest_activity = activity[:, NUM_ENTITY_SLOTS:]
        presence = torch.cat((entity["presence"], rest_activity), dim=1)
        road_observability = torch.cat(
            (
                entity["sector_visibility"],
                activity[:, NUM_ENTITY_SLOTS + 3 : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS],
            ),
            dim=1,
        )
        latent_observability = activity[:, -NUM_LATENT_SLOTS:]
        observability = torch.cat(
            (entity["observability"], road_observability, latent_observability), dim=1
        )
        rest_reliability = (
            observability[:, NUM_ENTITY_SLOTS:]
            * q_ground[:, NUM_ENTITY_SLOTS:]
            * q_view[:, NUM_ENTITY_SLOTS:]
        ).clamp(0.0, 1.0)
        reliability = torch.cat(
            (entity["entity_reliability"], rest_reliability), dim=1
        ).clamp(0.0, 1.0).detach()
        attributes = torch.cat(
            (
                presence.unsqueeze(-1),
                observability.unsqueeze(-1),
                horizontal,
                depth,
            ),
            dim=-1,
        )
        if q_view_sector is None:
            q_view_sector = torch.ones_like(entity["sector_visibility"])
        if (
            q_view_sector.shape != entity["sector_visibility"].shape
            or q_view_sector.requires_grad
        ):
            raise ValueError("dynamic q_view_sector must be detached [B,3]")
        absence = reliable_absence_evidence(
            entity["presence"],
            entity["horizontal_sector_probs"],
            entity["sector_visibility"],
            q_view_sector.to(
                device=entity["sector_visibility"].device,
                dtype=entity["sector_visibility"].dtype,
            ),
        )
        eps = torch.finfo(road_masks.dtype).eps
        boundary_logits = torch.logit(
            road_masks[:, 3:5].clamp(eps, 1.0 - eps)
        )
        return {
            "entity": entity,
            "road": {
                "drivable_logits": entity["drivable_logits"],
                "boundary_logits": boundary_logits,
                "boundary_style_logits": entity["boundary_style_logits"],
                "drivable_reliability": reliability[
                    :, NUM_ENTITY_SLOTS : NUM_ENTITY_SLOTS + 3
                ],
                "boundary_reliability": reliability[
                    :, NUM_ENTITY_SLOTS + 3 : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS
                ],
            },
            "presence": presence,
            "observability": observability,
            "reliability": reliability,
            "q_ground": q_ground.detach(),
            "q_view": q_view.detach(),
            "q_state": torch.cat(
                (
                    entity["q_state"],
                    torch.ones_like(q_ground[:, NUM_ENTITY_SLOTS:]),
                ),
                dim=1,
            ).detach(),
            "rho_clear": absence["clear_reliability"],
            "horizontal": horizontal,
            "depth": depth,
            "attributes": attributes,
            "absence": absence,
        }

    def _relations(
        self,
        target_tokens: Tensor,
        slot_tokens: Tensor,
        slot_masks: Tensor,
        slot_attributes: Mapping[str, Tensor],
        unary_module: RAELUnaryContribution,
        pairwise_module: RAELPairwiseContribution,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        unary = unary_module(
            target_tokens=target_tokens,
            evidence_tokens=slot_tokens,
            attributes=slot_attributes["attributes"],
            presence=slot_attributes["presence"],
            reliability=slot_attributes["reliability"],
        )
        pairwise = pairwise_module(
            target_tokens=target_tokens,
            evidence_tokens=slot_tokens,
            slot_masks=slot_masks,
            sector_probs=slot_attributes["horizontal"],
            unary_public_pi=unary["slot_weights"][..., :NUM_PUBLIC_SLOTS],
            reliability=slot_attributes["reliability"],
        )
        return unary, pairwise

    @staticmethod
    def _compose(global_logits: Tensor, unary: Mapping[str, Any], pairwise: Mapping[str, Any]) -> Tensor:
        return (
            global_logits.float()
            + unary["unary_contributions"].float().sum(dim=-1)
            + pairwise["pair_contributions"].float().sum(dim=-1)
        ).to(dtype=global_logits.dtype)

    def _variant_relations(
        self,
        action_tokens: Tensor,
        reason_tokens: Tensor,
        slot_tokens: Tensor,
        slot_masks: Tensor,
        slot_attributes: Mapping[str, Tensor],
        action_global: Tensor,
        reason_global: Tensor,
    ) -> tuple[Tensor, Tensor]:
        action_unary, action_pair = self._relations(
            action_tokens,
            slot_tokens,
            slot_masks,
            slot_attributes,
            self.action_unary,
            self.action_pairwise,
        )
        reason_unary, reason_pair = self._relations(
            reason_tokens,
            slot_tokens,
            slot_masks,
            slot_attributes,
            self.reason_unary,
            self.reason_pairwise,
        )
        return (
            self._compose(action_global, action_unary, action_pair),
            self._compose(reason_global, reason_unary, reason_pair),
        )

    def _pu_score_components(
        self,
        *,
        private_delta: Tensor,
        reason_public_pi: Tensor,
        reason_unary_raw: Tensor,
        reliability: Tensor,
        observability: Tensor,
    ) -> dict[str, Tensor]:
        """Expose detached P12 inputs without creating a second DINO view.

        The two views apply fixed feature-dropout masks to the P11 private
        residual only.  They therefore exercise an independent label-wise
        ``pu_private`` owner, preserve the action firewall, and remain
        deterministic for evaluator/audit reproducibility.
        """

        logits_one = self.pu_private_head(
            private_delta, self._pu_feature_keep_view_one
        )
        logits_two = self.pu_private_head(
            private_delta, self._pu_feature_keep_view_two
        )
        probs_one = torch.sigmoid(logits_one.float())
        probs_two = torch.sigmoid(logits_two.float())
        view_one = probs_one.detach()
        view_two = probs_two.detach()
        c_obs_input = (
            reason_public_pi.detach().float()
            * observability.detach().float().unsqueeze(1)
        ).sum(dim=-1).clamp(0.0, 1.0)
        confidence = reason_confidence_weights(
            reason_public_pi,
            reliability,
            reason_unary_raw,
            view_one,
            view_two,
            c_obs_input,
        )
        private_probability = (view_one * view_two).clamp_min(0.0).sqrt().detach()
        # P12 owns the formal soft-target equation.  Supplying zero observed
        # labels and zero lambda exposes its exact detached score only; no
        # dataset labels enter the formal representation forward.
        score_result = build_pu_soft_targets(
            torch.zeros_like(confidence["c_evidence"]),
            confidence["c_evidence"],
            private_probability,
            confidence["c_view"],
            confidence["c_obs"],
            torch.zeros(NUM_REASONS, device=private_delta.device, dtype=torch.float32),
            update_index=0,
        )
        return {
            "p_evidence": confidence["c_evidence"].detach(),
            "p_private": private_probability,
            "p_private_view_one": view_one,
            "p_private_view_two": view_two,
            "private_logits_view_one": logits_one,
            "private_logits_view_two": logits_two,
            "private_probs_view_one": probs_one,
            "private_probs_view_two": probs_two,
            "c_view": confidence["c_view"].detach(),
            "c_obs": confidence["c_obs"].detach(),
            "score": score_result["pu_score"].detach(),
        }

    def _branch_logits(
        self,
        *,
        requested: frozenset[str],
        action_visual: Tensor,
        action_tokens: Tensor,
        action_global: Tensor,
        reason_semantic: Tensor,
        reason_tokens: Tensor,
        reason_private_delta: Tensor,
        reason_global: Tensor,
        slot_tokens: Tensor,
        slot_masks: Tensor,
        slot_attributes: Mapping[str, Tensor],
        action_unary: Mapping[str, Any],
        action_pair: Mapping[str, Any],
        reason_unary: Mapping[str, Any],
        reason_pair: Mapping[str, Any],
        action_final: Tensor,
        reason_final: Tensor,
        global_context: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        candidates: dict[str, dict[str, Tensor]] = {
            "full": {"action": action_final, "reason": reason_final},
        }

        need_visual_global = "global_only" in requested
        action_visual_global: Tensor | None = None
        if need_visual_global:
            action_visual_global = self.action_category.project_global(action_visual)
        need_semantic_global = bool(
            requested.intersection({"global_only", "global_plus_semantic_bridge"})
        )
        reason_semantic_global: Tensor | None = None
        if need_semantic_global:
            reason_semantic_global = self.reason_private.reason_global_head(reason_semantic)
        if "global_only" in requested:
            assert action_visual_global is not None and reason_semantic_global is not None
            candidates["global_only"] = {
                "action": action_visual_global,
                "reason": reason_semantic_global,
            }
        if "global_plus_semantic_bridge" in requested:
            assert reason_semantic_global is not None
            candidates["global_plus_semantic_bridge"] = {
                "action": action_global,
                "reason": reason_semantic_global,
            }

        need_unary = bool(requested.intersection({"unary_only", "pairwise_off"}))
        if need_unary:
            action_unary_logits = (
                action_global + action_unary["unary_contributions"].sum(-1)
            )
            reason_unary_logits = (
                reason_global + reason_unary["unary_contributions"].sum(-1)
            )
            if "unary_only" in requested:
                candidates["unary_only"] = {
                    "action": action_unary_logits,
                    "reason": reason_unary_logits,
                }
            if "pairwise_off" in requested:
                candidates["pairwise_off"] = {
                    "action": action_unary_logits,
                    "reason": reason_unary_logits,
                }
        if "pairwise_only" in requested:
            candidates["pairwise_only"] = {
                "action": action_global + action_pair["pair_contributions"].sum(-1),
                "reason": reason_global + reason_pair["pair_contributions"].sum(-1),
            }
        if "pu_off" in requested:
            # PU changes training targets only.  At one fixed checkpoint this
            # raw representation is definitionally identical to full.
            candidates["pu_off"] = {
                "action": action_final,
                "reason": reason_final,
            }
        if "global_context_only" in requested:
            candidates["global_context_only"] = {
                "action": self.action_category.project_global(
                    global_context.unsqueeze(1).expand(-1, NUM_ACTIONS, -1)
                ),
                "reason": self.reason_private.reason_global_head(
                    global_context.unsqueeze(1).expand(-1, NUM_REASONS, -1)
                ),
            }

        if requested.intersection({"named_slots_only", "latent_slots_only"}):
            pair_indices = action_pair["pair_indices"]
            left, right = pair_indices.unbind(dim=-1)
            named_slot_mask = (
                torch.arange(NUM_PUBLIC_SLOTS, device=action_final.device) < 17
            )
            latent_slot_mask = ~named_slot_mask
            named_pair_mask = named_slot_mask[left] & named_slot_mask[right]
            latent_pair_mask = latent_slot_mask[left] | latent_slot_mask[right]
            if "named_slots_only" in requested:
                candidates["named_slots_only"] = {
                    "action": (
                        action_global
                        + action_unary["unary_contributions"][
                            ..., named_slot_mask
                        ].sum(-1)
                        + action_pair["pair_contributions"][
                            ..., named_pair_mask
                        ].sum(-1)
                    ),
                    "reason": (
                        reason_global
                        + reason_unary["unary_contributions"][
                            ..., named_slot_mask
                        ].sum(-1)
                        + reason_pair["pair_contributions"][
                            ..., named_pair_mask
                        ].sum(-1)
                    ),
                }
            if "latent_slots_only" in requested:
                candidates["latent_slots_only"] = {
                    "action": (
                        action_global
                        + action_unary["unary_contributions"][
                            ..., latent_slot_mask
                        ].sum(-1)
                        + action_pair["pair_contributions"][
                            ..., latent_pair_mask
                        ].sum(-1)
                    ),
                    "reason": (
                        reason_global
                        + reason_unary["unary_contributions"][
                            ..., latent_slot_mask
                        ].sum(-1)
                        + reason_pair["pair_contributions"][
                            ..., latent_pair_mask
                        ].sum(-1)
                    ),
                }

        if "semantic_reason_shuffled" in requested:
            semantic_shuffled = reason_semantic.roll(shifts=1, dims=1)
            shuffled_bridge = self.action_reason_bridge(
                action_visual, semantic_shuffled, self.action_category
            )
            shuffled_private = self.reason_private(
                semantic_shuffled,
                shuffled_bridge["action_bridged_tokens"],
                shuffled_bridge["z_A_global"],
            )
            shuffled_action, shuffled_reason = self._variant_relations(
                shuffled_bridge["action_bridged_tokens"],
                shuffled_private["formal_reason_tokens"],
                slot_tokens,
                slot_masks,
                slot_attributes,
                shuffled_bridge["z_A_global"],
                shuffled_private["z_R_global"],
            )
            candidates["semantic_reason_shuffled"] = {
                "action": shuffled_action,
                "reason": shuffled_reason,
            }

        if "no_semantic_reason" in requested:
            no_semantic = torch.zeros_like(reason_semantic)
            no_semantic_bridge = self.action_reason_bridge(
                action_visual, no_semantic, self.action_category
            )
            no_semantic_private = self.reason_private(
                no_semantic,
                no_semantic_bridge["action_bridged_tokens"],
                no_semantic_bridge["z_A_global"],
            )
            no_semantic_action, no_semantic_reason = self._variant_relations(
                no_semantic_bridge["action_bridged_tokens"],
                no_semantic_private["formal_reason_tokens"],
                slot_tokens,
                slot_masks,
                slot_attributes,
                no_semantic_bridge["z_A_global"],
                no_semantic_private["z_R_global"],
            )
            candidates["no_semantic_reason"] = {
                "action": no_semantic_action,
                "reason": no_semantic_reason,
            }

        if "reason_private_shuffled" in requested:
            private_shuffled_tokens = (
                reason_tokens - reason_private_delta + reason_private_delta.roll(1, dims=1)
            )
            private_shuffled_global = self.reason_private.reason_global_head(
                private_shuffled_tokens
            )
            private_unary, private_pair = self._relations(
                private_shuffled_tokens,
                slot_tokens,
                slot_masks,
                slot_attributes,
                self.reason_unary,
                self.reason_pairwise,
            )
            private_shuffled_reason = self._compose(
                private_shuffled_global,
                private_unary,
                private_pair,
            )
            candidates["reason_private_shuffled"] = {
                "action": action_final,
                "reason": private_shuffled_reason,
            }

        if "evidence_shuffled" in requested:
            reverse = torch.arange(
                NUM_PUBLIC_SLOTS - 1, -1, -1, device=slot_tokens.device
            )
            # Shuffle content only while preserving slot identity, geometry,
            # reliability, and routing attributes.  Jointly permuting all
            # fields would be relation-equivariant and therefore a false audit.
            evidence_shuffle_action, evidence_shuffle_reason = self._variant_relations(
                action_tokens,
                reason_tokens,
                slot_tokens.index_select(1, reverse),
                slot_masks,
                slot_attributes,
                action_global,
                reason_global,
            )
            candidates["evidence_shuffled"] = {
                "action": evidence_shuffle_action,
                "reason": evidence_shuffle_reason,
            }

        selected = {"full": candidates["full"]}
        for name in requested:
            selected[name] = candidates[name]
        return selected

    def _decode_from_field_impl(
        self,
        field: RAELVisualBundle,
        *,
        diagnostic_modes: tuple[str, ...],
        call_counter: _ForwardCallCounter,
        q_ground: Tensor | None = None,
        q_view: Tensor | None = None,
        q_view_sector: Tensor | None = None,
        finalize_collapse: bool = True,
    ) -> dict[str, Any]:
        self._validate_diagnostic_modes(diagnostic_modes)
        if not isinstance(field, RAELVisualBundle):
            raise TypeError("field must be an RAELVisualBundle from encode_images")
        prepared = field.prepared_field
        ledger = self.slot_ledger(prepared)
        public_evidence = ledger["public_evidence"]
        slot_tokens = ledger["slot_tokens"]
        ledger_slot_masks = ledger["slot_masks"]
        field_keys = prepared.get("keys_by_layer")
        if not torch.is_tensor(field_keys):
            raise ValueError("E04-R2 requires prepared keys_by_layer")
        slot_masks = self._canonical_slot_masks(slot_tokens, field_keys)
        slot_attributes = self._slot_attributes(
            evidence_slots=slot_tokens,
            canonical_masks=slot_masks,
            q_ground=q_ground,
            q_view=q_view,
            q_view_sector=q_view_sector,
        )
        canonical_geometry = self.slot_ledger.geometry_from_masks(slot_masks)

        semantic = self.semantic_reason(
            self.multilayer_field,
            prepared,
            self.slot_ledger.to_evidence_read_bundle(public_evidence),
        )
        action_foundation = self.action_category(self.multilayer_field, prepared)
        bridge = self.action_reason_bridge(
            action_foundation["action_visual_tokens"],
            semantic["semantic_reason_tokens"],
            self.action_category,
        )
        private = self.reason_private(
            semantic["semantic_reason_tokens"],
            bridge["action_bridged_tokens"],
            bridge["z_A_global"],
        )

        action_unary, action_pair = self._relations(
            bridge["action_bridged_tokens"],
            slot_tokens,
            slot_masks,
            slot_attributes,
            self.action_unary,
            self.action_pairwise,
        )
        reason_unary, reason_pair = self._relations(
            private["formal_reason_tokens"],
            slot_tokens,
            slot_masks,
            slot_attributes,
            self.reason_unary,
            self.reason_pairwise,
        )
        action_final = self._compose(bridge["z_A_global"], action_unary, action_pair)
        reason_final = self._compose(private["z_R_global"], reason_unary, reason_pair)
        reason_semantic_logits = self.reason_private.reason_global_head(
            semantic["semantic_reason_tokens"]
        )

        pair_indices = action_pair["pair_indices"]
        action_named_ratio, action_latent_ratio = _target_partition_ratios(
            action_unary["unary_contributions"],
            action_pair["pair_contributions"],
            pair_indices,
        )
        reason_named_ratio, reason_latent_ratio = _target_partition_ratios(
            reason_unary["unary_contributions"],
            reason_pair["pair_contributions"],
            pair_indices,
        )

        reason_public_pi = reason_unary["slot_weights"][..., :NUM_PUBLIC_SLOTS]
        pu = self._pu_score_components(
            private_delta=private["private_delta"],
            reason_public_pi=reason_public_pi,
            reason_unary_raw=reason_unary["unary_contributions_raw"],
            reliability=slot_attributes["reliability"],
            observability=slot_attributes["observability"],
        )

        requested_branches = frozenset(diagnostic_modes)
        branches = self._branch_logits(
            requested=requested_branches,
            action_visual=action_foundation["action_visual_tokens"],
            action_tokens=bridge["action_bridged_tokens"],
            action_global=bridge["z_A_global"],
            reason_semantic=semantic["semantic_reason_tokens"],
            reason_tokens=private["formal_reason_tokens"],
            reason_private_delta=private["private_delta"],
            reason_global=private["z_R_global"],
            slot_tokens=slot_tokens,
            slot_masks=slot_masks,
            slot_attributes=slot_attributes,
            action_unary=action_unary,
            action_pair=action_pair,
            reason_unary=reason_unary,
            reason_pair=reason_pair,
            action_final=action_final,
            reason_final=reason_final,
            global_context=ledger["global_context"],
        )

        internal = self.slot_ledger.audit_diagnostics(public_evidence)
        collapse = (
            self.multilayer_field.finalize_batch_collapse(
                prepared,
                {
                    "action": {
                        "layer_weights": action_foundation["layer_weights"],
                    },
                    "reason": {
                        "layer_weights": semantic["layer_weights"],
                    },
                    "slots": {
                        "layer_weights": internal.layer_weights_iteration2,
                    },
                },
            )
            if finalize_collapse
            else {"provisional": True, "updated": False}
        )
        action_reconstructed = (
            bridge["z_A_global"].float()
            + action_unary["unary_contributions"].float().sum(-1)
            + action_pair["pair_contributions"].float().sum(-1)
        )
        reason_reconstructed = (
            private["z_R_global"].float()
            + reason_unary["unary_contributions"].float().sum(-1)
            + reason_pair["pair_contributions"].float().sum(-1)
        )
        module_call_summary = call_counter.snapshot()
        action_positive, action_negative = _signed_partition(
            bridge["z_A_global"],
            action_unary["unary_contributions"],
            action_pair["pair_contributions"],
        )
        reason_positive, reason_negative = _signed_partition(
            private["z_R_global"],
            reason_unary["unary_contributions"],
            reason_pair["pair_contributions"],
        )
        absence = slot_attributes["absence"]
        latent_slots = slot_tokens[:, -NUM_LATENT_SLOTS:]
        latent_keep_probability = self._latent_feature_keep_view_one.float().mean()
        latent_feature_view_one = (
            latent_slots * self._latent_feature_keep_view_one
            / latent_keep_probability.clamp_min(1.0e-6)
        )
        latent_keep_probability = self._latent_feature_keep_view_two.float().mean()
        latent_feature_view_two = (
            latent_slots * self._latent_feature_keep_view_two
            / latent_keep_probability.clamp_min(1.0e-6)
        )
        result: dict[str, Any] = {
            "action_logits_visual_global": action_foundation["z_A_global_visual"],
            "action_logits_global": bridge["z_A_global"],
            "action_logits_final": action_final,
            "reason_logits_semantic_global": reason_semantic_logits,
            "reason_logits_global": private["z_R_global"],
            "reason_logits_final": reason_final,
            "reason_logits_semantic": reason_semantic_logits,
            "semantic_reason_tokens": semantic["semantic_reason_tokens"],
            "reason_tokens": private["formal_reason_tokens"],
            "reason_private_delta": private["private_delta"],
            "action_tokens": bridge["action_bridged_tokens"],
            "action_semantic_delta": (
                bridge["action_bridged_tokens"]
                - action_foundation["action_visual_tokens"]
            ),
            "evidence_slots": slot_tokens,
            # Legacy ledger masks are detached diagnostics and cannot enter a
            # grounding, relation, or counterfactual formal loss path.
            "ledger_slot_masks_diagnostic": ledger_slot_masks.detach(),
            "ledger_background_mask_diagnostic": ledger["background_mask"].detach(),
            # P14 captures these as fixed replay metadata.  They preserve the
            # numerical relation context but cannot create a second gradient
            # path around the P16 evidence/token admission boundaries.
            "counterfactual_slot_attributes": {
                "attributes": slot_attributes["attributes"].detach(),
                "presence": slot_attributes["presence"].detach(),
                "reliability": slot_attributes["reliability"].detach(),
                "horizontal": slot_attributes["horizontal"].detach(),
            },
            "grounding_outputs": {
                "entity": {
                    "presence_logits": slot_attributes["entity"]["presence_logits"],
                    "entity_type_logits": slot_attributes["entity"][
                        "entity_type_logits"
                    ],
                    "traffic_state_logits": slot_attributes["entity"][
                        "traffic_state_logits"
                    ],
                    "entity_reliability": slot_attributes["entity"][
                        "entity_reliability"
                    ],
                },
                "road": slot_attributes["road"],
            },
            "entity_slots": slot_tokens[:, :NUM_ENTITY_SLOTS],
            "road_slots": slot_tokens[
                :, NUM_ENTITY_SLOTS : NUM_ENTITY_SLOTS + NUM_ROAD_SLOTS
            ],
            "latent_slots": latent_slots,
            "latent_feature_view_one": latent_feature_view_one,
            "latent_feature_view_two": latent_feature_view_two,
            "background_slot": ledger["background_token"].detach().unsqueeze(1),
            # slot_masks is the E04-R2 canonical admission-dominated mask.
            "slot_masks": slot_masks,
            "background_mask": ledger["background_mask"].detach(),
            "slot_area": canonical_geometry["area"],
            "slot_centroid": canonical_geometry["centroid"],
            "slot_scale": canonical_geometry["scale"],
            "slot_activity": canonical_geometry["activity"],
            "slot_presence": slot_attributes["presence"],
            "slot_observability": slot_attributes["observability"],
            "slot_type_probs": slot_attributes["entity"]["entity_type_probs"],
            "slot_state_probs": slot_attributes["entity"]["traffic_state_probs"],
            "slot_sector_probs": {
                "horizontal": slot_attributes["horizontal"],
                "depth": slot_attributes["depth"],
            },
            "slot_reliability": slot_attributes["reliability"],
            "slot_q_ground": slot_attributes["q_ground"],
            "slot_q_view": slot_attributes["q_view"],
            "slot_q_state": slot_attributes["q_state"],
            "road_rho_clear": slot_attributes["rho_clear"],
            "action_slot_weights": action_unary["slot_weights"],
            "reason_slot_weights": reason_unary["slot_weights"],
            "action_unary_contributions": action_unary["unary_contributions"],
            "reason_unary_contributions": reason_unary["unary_contributions"],
            "action_pairwise_contributions": action_pair["pair_contributions"],
            "reason_pairwise_contributions": reason_pair["pair_contributions"],
            "action_pairwise_incident_contributions": action_pair[
                "incident_postgamma_by_slot"
            ],
            "reason_pairwise_incident_contributions": reason_pair[
                "incident_postgamma_by_slot"
            ],
            "action_analytical_deletion": (
                action_unary["unary_contributions"]
                + action_pair["incident_postgamma_by_slot"]
            ),
            "reason_analytical_deletion": (
                reason_unary["unary_contributions"]
                + reason_pair["incident_postgamma_by_slot"]
            ),
            "action_pair_indices": action_pair["pair_indices"],
            "reason_pair_indices": reason_pair["pair_indices"],
            "action_unary_contributions_raw": action_unary[
                "unary_contributions_raw"
            ],
            "reason_unary_contributions_raw": reason_unary[
                "unary_contributions_raw"
            ],
            "action_pairwise_contributions_raw": action_pair[
                "pair_contributions_raw"
            ],
            "reason_pairwise_contributions_raw": reason_pair[
                "pair_contributions_raw"
            ],
            "action_global_contribution": bridge["z_A_global"],
            "reason_global_contribution": private["z_R_global"],
            "named_contribution_ratio": {
                "action": action_named_ratio,
                "reason": reason_named_ratio,
            },
            "latent_contribution_ratio": {
                "action": action_latent_ratio,
                "reason": reason_latent_ratio,
            },
            "positive_contribution": {
                "action": action_positive,
                "reason": reason_positive,
            },
            "negative_contribution": {
                "action": action_negative,
                "reason": reason_negative,
            },
            "null_mass": {
                "action": action_unary["null_mass"],
                "reason": reason_unary["null_mass"],
            },
            "layer_weights_action": action_foundation["layer_weights"],
            "layer_weights_reason": semantic["layer_weights"],
            "layer_weights_slots": internal.layer_weights_iteration2,
            "clear_left": absence["clear"][:, 0],
            "clear_center": absence["clear"][:, 1],
            "clear_right": absence["clear"][:, 2],
            "occupied_left": absence["occupied"][:, 0],
            "occupied_center": absence["occupied"][:, 1],
            "occupied_right": absence["occupied"][:, 2],
            "pu_scores": pu["score"],
            "pu_active_labels": self._pu_active_labels.clone(),
            "branch_logits": branches,
            "diagnostics": {
                "dino_call_count": int(field.dino_outputs["dino_call_count"]),
                "module_call_summary": module_call_summary,
                "action_reconstruction_max_error": (
                    action_reconstructed - action_final.float()
                ).abs().amax().detach(),
                "reason_reconstruction_max_error": (
                    reason_reconstructed - reason_final.float()
                ).abs().amax().detach(),
                "reason_private_in_action_graph": False,
                "background_allow_contribution": False,
                "collapse": collapse,
                "pu": pu,
                "bridge": bridge["diagnostics"],
                "reason_private": private["diagnostics"],
                "action_unary": action_unary["diagnostics"],
                "reason_unary": reason_unary["diagnostics"],
                "action_pairwise": action_pair["diagnostics"],
                "reason_pairwise": reason_pair["diagnostics"],
                "internal_slot_count": 21,
                "public_slot_count": 20,
                "requested_diagnostic_modes": tuple(diagnostic_modes),
            },
        }
        return result

    def _counterfactual_shared_field(self, field: RAELVisualBundle) -> Tensor:
        prepared = field.prepared_field
        field_tokens = prepared.get("values_by_layer")
        grid_hw = prepared.get("grid_hw")
        if not torch.is_tensor(field_tokens) or field_tokens.ndim != 4:
            raise ValueError("counterfactual replay requires values_by_layer [B,4,N,D]")
        if tuple(field_tokens.shape[1:3]) != (4, 45 * 80) or field_tokens.shape[-1] != self.dim:
            raise ValueError("counterfactual replay requires the formal [B,4,3600,384] field")
        if tuple(grid_hw) != (45, 80):
            raise ValueError("counterfactual replay requires grid_hw=(45,80)")
        batch = field_tokens.shape[0]
        return (
            field_tokens.permute(0, 1, 3, 2)
            .reshape(batch, 4 * self.dim, 45, 80)
        )

    def _counterfactual_values_from_field(
        self,
        field: RAELVisualBundle,
        shared_field: Tensor,
    ) -> Tensor:
        expected = (
            field.prepared_field["values_by_layer"].shape[0],
            4 * self.dim,
            45,
            80,
        )
        if shared_field.shape != expected:
            raise ValueError(f"counterfactual shared_field must be {expected}")
        batch = shared_field.shape[0]
        field_tokens = (
            shared_field.reshape(batch, 4, self.dim, 45 * 80)
            .permute(0, 1, 3, 2)
        )
        return field_tokens

    @staticmethod
    def _counterfactual_slot_readout(
        values_by_layer: Tensor,
        slot_masks: Tensor,
        layer_weights: Tensor,
    ) -> Tensor:
        """Pool frozen second-route assignments from a supplied value field.

        E08-R3 keeps competition, masks, and layer routes fixed.  This helper
        is therefore only the value-pooling part of the residual-anchored
        second-readout operator; it never updates a slot assignment.
        """

        if values_by_layer.ndim != 4 or values_by_layer.shape[1:] != (4, 45 * 80, DIM):
            raise ValueError("counterfactual values must be [B,4,3600,384]")
        batch = values_by_layer.shape[0]
        if slot_masks.shape != (batch, NUM_PUBLIC_SLOTS, 45, 80):
            raise ValueError("counterfactual slot_masks must be [B,20,45,80]")
        if layer_weights.shape != (batch, NUM_PUBLIC_SLOTS, 4):
            raise ValueError("counterfactual layer_weights must be [B,20,4]")
        masks = slot_masks.reshape(batch, NUM_PUBLIC_SLOTS, 45 * 80)
        denominator = masks.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        pooled = torch.zeros(
            batch,
            NUM_PUBLIC_SLOTS,
            DIM,
            device=values_by_layer.device,
            dtype=values_by_layer.dtype,
        )
        for layer_index in range(4):
            selected = torch.einsum(
                "bjn,bnd->bjd", masks, values_by_layer[:, layer_index]
            )
            pooled = pooled + layer_weights[:, :, layer_index].unsqueeze(-1) * selected
        return pooled / denominator

    def _counterfactual_second_readout(
        self,
        evidence_slots: Tensor,
        values_by_layer: Tensor,
        slot_masks: Tensor,
        layer_weights: Tensor,
    ) -> Tensor:
        """Run only frozen ledger iteration-two pooling and GRU readout.

        This is R_sg(thetaL)(E, sg(V), sg(C)) from E08-R3.  It deliberately
        does not call the ledger, competition, visual router, or mask updater.
        The GRU parameters are detached constants while E remains connected,
        so the returned Jacobian with respect to E is the real readout
        Jacobian instead of a replay-owner or straight-through gradient.
        """

        if evidence_slots.ndim != 3 or evidence_slots.shape[1:] != (
            NUM_PUBLIC_SLOTS,
            self.dim,
        ):
            raise ValueError("counterfactual evidence_slots must be [B,20,384]")
        if values_by_layer.shape[0] != evidence_slots.shape[0]:
            raise ValueError("counterfactual values and evidence batch sizes must match")
        pooled = self._counterfactual_slot_readout(
            values_by_layer,
            slot_masks,
            layer_weights,
        )
        frozen_state = {
            name: parameter.detach()
            for name, parameter in self.slot_ledger.slot_gru.named_parameters()
        }
        frozen_state.update(
            {
                name: buffer.detach()
                for name, buffer in self.slot_ledger.slot_gru.named_buffers()
            }
        )
        return functional_call(
            self.slot_ledger.slot_gru,
            frozen_state,
            args=(
                pooled.reshape(-1, self.dim),
                evidence_slots.reshape(-1, self.dim),
            ),
            strict=True,
        ).reshape_as(evidence_slots)

    @staticmethod
    def _frozen_contribution_call(
        module: nn.Module,
        **kwargs: Tensor,
    ) -> Mapping[str, Any]:
        """Evaluate contribution ownership as a fixed differentiable function.

        Its parameters are detached constants, so CF gradients reach only the
        formal P16 token/evidence inputs.  This avoids a replay-only optimizer
        route around P13 without inventing a straight-through estimator.
        """

        frozen_state = {
            name: parameter.detach()
            for name, parameter in module.named_parameters()
        }
        frozen_state.update(
            {
                name: buffer.detach()
                for name, buffer in module.named_buffers()
            }
        )
        return functional_call(module, frozen_state, args=(), kwargs=kwargs, strict=True)

    def build_counterfactual_replay(
        self,
        field: RAELVisualBundle,
        outputs: Mapping[str, Any],
        *,
        target_family: str,
    ) -> dict[str, Tensor | Callable[[Tensor], Tensor]]:
        """Build a P14 replay over formal P16 boundaries.

        The replay receives real selected/control replacements in the shared
        four-layer value field.  It recomputes only the fixed-mask slot
        readout delta and frozen-parameter contribution values; DINO and the
        two-round ledger are never re-run.  Gradients therefore enter through
        original ``evidence_slots`` and action/semantic token boundaries for
        P13 admission, not through replay-only owners.
        """

        if not isinstance(field, RAELVisualBundle):
            raise TypeError("field must come from encode_images")
        if target_family not in {"action", "reason"}:
            raise ValueError("target_family must be 'action' or 'reason'")
        if target_family == "action":
            # E08-R3 freezes the action target branch.  Action CF may enter
            # only the original evidence boundary, never the semantic bridge.
            target_tokens = outputs["action_tokens"].detach()
            global_logits = outputs["action_logits_global"].detach()
            unary_module = self.action_unary
            pairwise_module = self.action_pairwise
        else:
            # Keep the formal reason input numerically exact while making the
            # P16 semantic token boundary the only trainable CF target route.
            target_tokens = (
                outputs["semantic_reason_tokens"]
                + outputs["reason_private_delta"].detach()
            )
            global_logits = outputs["reason_logits_global"].detach()
            unary_module = self.reason_unary
            pairwise_module = self.reason_pairwise

        # E08-R3 freezes all visual-side replay inputs.  Gradients may only
        # enter the original P16 evidence/token boundaries captured below.
        shared_field = self._counterfactual_shared_field(field).detach()
        original_slots = outputs.get("evidence_slots")
        original_masks = outputs.get("slot_masks")
        original_layer_weights = outputs.get("layer_weights_slots")
        relation_attributes = outputs.get("counterfactual_slot_attributes")
        if not torch.is_tensor(original_slots) or original_slots.shape != (
            shared_field.shape[0], NUM_PUBLIC_SLOTS, self.dim
        ):
            raise ValueError("counterfactual replay requires formal evidence_slots [B,20,384]")
        if not torch.is_tensor(original_masks) or original_masks.shape != (
            shared_field.shape[0], NUM_PUBLIC_SLOTS, 45, 80
        ):
            raise ValueError("counterfactual replay requires formal slot_masks [B,20,45,80]")
        if not torch.is_tensor(original_layer_weights) or original_layer_weights.shape != (
            shared_field.shape[0], 21, 4
        ):
            raise ValueError("counterfactual replay requires formal layer_weights_slots [B,21,4]")
        if not isinstance(relation_attributes, Mapping):
            raise ValueError("counterfactual replay requires fixed slot attributes")
        required_attributes = ("attributes", "presence", "reliability", "horizontal")
        if any(not torch.is_tensor(relation_attributes.get(name)) for name in required_attributes):
            raise ValueError("counterfactual replay slot attributes are incomplete")
        static_attributes = {
            name: relation_attributes[name].detach()
            for name in required_attributes
        }
        fixed_masks = original_masks.detach()
        fixed_layer_weights = original_layer_weights[:, :NUM_PUBLIC_SLOTS].detach()

        def second_readout(replay_field: Tensor) -> Tensor:
            # V and C are stop-gradient intervention context.  E is retained
            # deliberately so R0/RI supply the exact E08-R3 Jacobian.
            return self._counterfactual_second_readout(
                original_slots,
                self._counterfactual_values_from_field(field, replay_field).detach(),
                fixed_masks,
                fixed_layer_weights,
            )

        r0 = second_readout(shared_field)

        def intervened_evidence(replay_field: Tensor) -> Tensor:
            # The identity branch is mathematically EI=E+(R0-R0), but returning
            # E directly preserves bitwise identity and its exact identity
            # Jacobian without a surrogate branch.
            if replay_field is shared_field:
                return original_slots
            ri = second_readout(replay_field)
            return original_slots + (ri - r0)

        cache: dict[str, Any] = {"field": None}

        def replay(replay_field: Tensor) -> tuple[Tensor, Tensor]:
            if cache["field"] is replay_field:
                return cache["readout"], cache["contribution"]
            replay_slots = intervened_evidence(replay_field)
            unary = self._frozen_contribution_call(
                unary_module,
                target_tokens=target_tokens,
                evidence_tokens=replay_slots,
                attributes=static_attributes["attributes"],
                presence=static_attributes["presence"],
                reliability=static_attributes["reliability"],
            )
            pairwise = self._frozen_contribution_call(
                pairwise_module,
                target_tokens=target_tokens,
                evidence_tokens=replay_slots,
                slot_masks=fixed_masks,
                sector_probs=static_attributes["horizontal"],
                unary_public_pi=unary["slot_weights"][..., :NUM_PUBLIC_SLOTS],
                reliability=static_attributes["reliability"],
            )
            readout = global_logits + unary["unary_contributions"].sum(dim=-1)
            contribution = pairwise["pair_contributions"].sum(dim=-1)
            cache.update(
                {
                    "field": replay_field,
                    "readout": readout,
                    "contribution": contribution,
                }
            )
            return readout, contribution

        return {
            "shared_field": shared_field,
            "second_readout": second_readout,
            "intervened_evidence": intervened_evidence,
            "public_readout": lambda replay_field: replay(replay_field)[0],
            "public_contribution": lambda replay_field: replay(replay_field)[1],
        }

    def decode_from_field(
        self,
        field: RAELVisualBundle,
        *,
        diagnostic_modes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        self._validate_diagnostic_modes(diagnostic_modes)
        with _ForwardCallCounter(self._counted_modules()) as call_counter:
            return self._decode_from_field_impl(
                field,
                diagnostic_modes=diagnostic_modes,
                call_counter=call_counter,
            )

    def decode_from_field_provisional(
        self,
        field: RAELVisualBundle,
    ) -> dict[str, Any]:
        """Decode current slots for dynamic matching without consuming batch state."""

        with _ForwardCallCounter(self._counted_modules()) as call_counter:
            return self._decode_from_field_impl(
                field,
                diagnostic_modes=(),
                call_counter=call_counter,
                finalize_collapse=False,
            )

    def decode_from_field_with_reliability(
        self,
        field: RAELVisualBundle,
        *,
        q_ground: Tensor,
        q_view: Tensor,
        q_view_sector: Tensor,
        diagnostic_modes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Re-run task readout from one encoded field with detached dynamic reliability."""

        self._validate_diagnostic_modes(diagnostic_modes)
        for name, value in (
            ("q_ground", q_ground),
            ("q_view", q_view),
            ("q_view_sector", q_view_sector),
        ):
            if not isinstance(value, Tensor) or value.requires_grad:
                raise ValueError(f"{name} must be a detached tensor")
        with _ForwardCallCounter(self._counted_modules()) as call_counter:
            return self._decode_from_field_impl(
                field,
                diagnostic_modes=diagnostic_modes,
                call_counter=call_counter,
                q_ground=q_ground,
                q_view=q_view,
                q_view_sector=q_view_sector,
            )

    def forward(
        self,
        images: Tensor,
        *,
        diagnostic_modes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        self._validate_diagnostic_modes(diagnostic_modes)
        with _ForwardCallCounter(self._counted_modules()) as call_counter:
            field = self._encode_images_impl(images, call_counter=call_counter)
            return self._decode_from_field_impl(
                field,
                diagnostic_modes=diagnostic_modes,
                call_counter=call_counter,
            )


__all__ = ["BRANCH_NAMES", "RAELOIAModel", "RAELVisualBundle"]
