from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .tida_action_motion_cross_attention import TIDAActionMotionCrossAttention
from .tida_action_reader import TIDAActionReader
from .tida_context_encoder import TIDAContextEncoder
from .tida_flow_transition_bank import TIDAFlowTransitionBank
from .tida_geometric_flow import TIDAGeometricFlowDecisionHeads, TIDAGeometricFlowEncoder
from .tida_predicate_differential import TIDAPredicateDifferential
from .tida_reason_reader import TIDAReasonReader
from .tida_relational_traffic_flow import (
    TIDARelationalTrafficFlow,
    select_semantic_traffic_seeds,
)
from .tida_object_intent_flow import TIDAObjectIntentTransport
from .tida_object_tracker import TIDAFrozenPointTracker
from .tida_temporal_encoder import TIDATemporalEncoder
from .tida_terminal_innovation import TIDATerminalInnovation
from .tida_terminal_query_reader import TIDATerminalQueryReader
from .tida_traffic_trajectories import TIDATrafficTrajectoryBuilder
from .tida_traffic_trajectory_head import TIDATrafficTrajectoryHead
from .tida_traffic_boundary import TIDATrafficAdaptiveBoundary
from ..explain.tida_dynamic_concepts import translate_dynamic_concepts


def confidence_aware_reason_delta(
    image_logits: torch.Tensor,
    temporal_delta: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Keep positive temporal evidence while protecting likely image positives."""
    if image_logits.shape != temporal_delta.shape:
        raise ValueError("image logits and temporal delta must have identical shapes")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    positive_delta = torch.relu(temporal_delta)
    negative_delta = torch.relu(-temporal_delta)
    negative_permission = 1.0 - torch.sigmoid(image_logits.detach() / float(temperature))
    return positive_delta - negative_permission * negative_delta


class TIDAFrozenVETRAImageBase(nn.Module):
    """Expose a Stage-A/Stage-B VETRA checkpoint as one frozen image model."""

    def __init__(
        self,
        image_model: nn.Module,
        refiner: nn.Module | None = None,
        *,
        action_scale: float = 1.0,
        reason_scale: float = 0.60,
    ) -> None:
        super().__init__()
        self.image_model = image_model
        self.refiner = refiner
        self.action_scale = float(action_scale)
        self.reason_scale = float(reason_scale)
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @property
    def foundation(self):
        return self.image_model.foundation

    def encode_images(self, images: torch.Tensor) -> dict[str, Any]:
        return self.image_model.encode_images(images)

    def decode_from_field(self, field: dict[str, Any], **_: Any) -> dict[str, Any]:
        source = self.image_model.decode_from_field(
            field, action_scale=self.action_scale, reason_scale=self.reason_scale
        )
        if self.refiner is None:
            return source
        from .vetra_strong_refiner import SelectiveActionPathRefiner, SelectiveVisualActionRankRefiner

        if isinstance(self.refiner, SelectiveVisualActionRankRefiner):
            refined = self.refiner(
                source["action_logits_final"], source["reason_logits_final"],
                source["action_nodes_primary"], source["evidence_token"],
            )
        elif isinstance(self.refiner, SelectiveActionPathRefiner):
            refined = self.refiner(source, action_scale=self.action_scale)
        else:
            raise TypeError(f"unsupported frozen VETRA refiner: {type(self.refiner).__name__}")
        if not torch.equal(refined["reason_logits_final"], source["reason_logits_final"]):
            raise RuntimeError("VETRA Stage-B refiner changed reason logits")
        return {
            **source,
            "action_logits_pre_stage_b": source["action_logits_final"],
            "action_logits_final": refined["action_logits_final"],
            "stage_b_action_delta": refined["action_delta"],
        }

    def train(self, mode: bool = True):
        super().train(False)
        self.image_model.eval()
        if self.refiner is not None:
            self.refiner.eval()
        return self


class TIDAOIAModel(nn.Module):
    """Frozen image OIA plus target-conditioned temporal innovation."""

    OWNER_MODULES = {
        "history_reader": "query_reader",
        "temporal_encoder": "temporal_encoder",
        "innovation_predictor": "terminal_innovation",
        "predicate_differential": "predicate_differential",
        "flow_transition": "flow_transition_bank",
        "temporal_action": "action_reader",
        "temporal_reason": "reason_reader",
    }

    def __init__(
        self,
        image_model: nn.Module,
        *,
        dim: int = 384,
        num_actions: int = 4,
        num_reasons: int = 21,
        num_predicates: int = 32,
        predicate_roles: dict[str, list[str]] | None = None,
        predicate_role_path: str = "configs/tida_predicate_roles.yaml",
        context_chunk_size: int = 2,
        action_evidence_trust_cap: float = 0.25,
        reason_evidence_trust_cap: float = 0.25,
        conditional_temporal_utility: bool = False,
        action_temporal_budget_cap: float = 0.60,
        reason_temporal_budget_cap: float = 0.50,
        confidence_aware_reason_gate: bool = False,
        reason_gate_temperature: float = 0.5,
        geometric_flow_enabled: bool = False,
        geometric_flow_hidden_dim: int = 64,
        geometric_action_cap: float = 0.20,
        geometric_reason_cap: float = 0.15,
        traffic_action_enabled: bool = False,
        traffic_action_cap: float = 0.15,
        traffic_motion_topk: int = 12,
        traffic_trajectory_enabled: bool = False,
        traffic_trajectory_cap: float = 0.08,
        traffic_trajectory_heads: int = 4,
        traffic_trajectory_state_enabled: bool = True,
        traffic_trajectory_state_strength_scale: float = 8.0,
        traffic_trajectory_state_cap_ratio: float = 1.0,
        traffic_trajectory_state_utility_open_prior: float = 0.10,
        relational_traffic_enabled: bool = False,
        relational_traffic_action_cap: float = 0.12,
        relational_traffic_reason_cap: float = 0.10,
        relational_traffic_heads: int = 4,
        relational_traffic_reason_indices: tuple[int, ...] | None = None,
        traffic_adaptive_boundary_enabled: bool = False,
        traffic_adaptive_boundary_cap: float = 0.25,
        object_intent_enabled: bool = False,
        object_tracker: nn.Module | None = None,
        object_tracker_repository: str = r"E:\sbw\deps\co-tracker",
        object_tracker_model_name: str = "cotracker3_offline",
        object_tracker_grid_size: int = 8,
        object_intent_action_cap: float = 0.08,
        object_intent_reason_cap: float = 0.06,
        object_intent_heads: int = 4,
        object_intent_reason_indices: tuple[int, ...] | None = None,
        object_intent_role_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        self.image_model = image_model
        for parameter in self.image_model.parameters():
            parameter.requires_grad = False
        self.image_model.eval()
        predicate_names = list(self.image_model.foundation.predicate_head.names)
        if len(predicate_names) != num_predicates:
            raise ValueError(f"TIDA requires exactly {num_predicates} predicates")
        selected_layers = tuple(self.image_model.foundation.dino.selected_layers)
        self.query_reader = TIDATerminalQueryReader(dim, num_actions, num_predicates, selected_layers)
        self.context_encoder = TIDAContextEncoder(
            self.image_model.foundation.dino, self.query_reader,
            context_chunk_size=context_chunk_size, motion_topk=traffic_motion_topk,
        )
        self.temporal_encoder = TIDATemporalEncoder(dim=dim, num_layers=2, num_heads=4, dropout=0.10)
        self.terminal_innovation = TIDATerminalInnovation(dim=dim)
        self.predicate_differential = TIDAPredicateDifferential(
            dim=dim,
            predicate_names=predicate_names,
            roles=predicate_roles,
            role_path=predicate_role_path,
        )
        self.flow_transition_bank = TIDAFlowTransitionBank(dim=dim, region_count=5)
        self.action_reader = TIDAActionReader(
            dim,
            num_actions,
            num_predicates,
            kappa=0.15,
            evidence_trust_cap=action_evidence_trust_cap,
            conditional_utility_enabled=conditional_temporal_utility,
            conditional_flow_mix_cap=action_temporal_budget_cap,
        )
        self.reason_reader = TIDAReasonReader(
            dim,
            num_reasons,
            kappa=0.12,
            evidence_trust_cap=reason_evidence_trust_cap,
            conditional_utility_enabled=conditional_temporal_utility,
            conditional_flow_mix_cap=reason_temporal_budget_cap,
        )
        self.confidence_aware_reason_gate = bool(confidence_aware_reason_gate)
        self.reason_gate_temperature = float(reason_gate_temperature)
        self.geometric_flow_enabled = bool(geometric_flow_enabled)
        self.geometric_flow = TIDAGeometricFlowEncoder(hidden_dim=geometric_flow_hidden_dim)
        self.geometric_heads = TIDAGeometricFlowDecisionHeads(
            hidden_dim=geometric_flow_hidden_dim,
            num_actions=num_actions,
            num_reasons=num_reasons,
            action_cap=geometric_action_cap,
            reason_cap=geometric_reason_cap,
        )
        if not self.geometric_flow_enabled:
            for parameter in self.geometric_heads.parameters():
                parameter.requires_grad = False
        self.traffic_action_enabled = bool(traffic_action_enabled)
        self.traffic_action = TIDAActionMotionCrossAttention(
            dim=dim, num_actions=num_actions, cap=traffic_action_cap
        )
        if not self.traffic_action_enabled:
            for parameter in self.traffic_action.parameters():
                parameter.requires_grad = False
        self.traffic_trajectory_enabled = bool(traffic_trajectory_enabled)
        self.traffic_trajectory_builder = TIDATrafficTrajectoryBuilder()
        self.traffic_trajectory_head = TIDATrafficTrajectoryHead(
            dim=dim, num_actions=num_actions, num_heads=traffic_trajectory_heads,
            cap=traffic_trajectory_cap,
            state_enabled=traffic_trajectory_state_enabled,
            state_strength_scale=traffic_trajectory_state_strength_scale,
            state_cap_ratio=traffic_trajectory_state_cap_ratio,
            state_utility_open_prior=traffic_trajectory_state_utility_open_prior,
        )
        self.relational_traffic_enabled = bool(relational_traffic_enabled)
        self.relational_traffic = TIDARelationalTrafficFlow(
            dim=dim,
            num_actions=num_actions,
            num_reasons=num_reasons,
            heads=relational_traffic_heads,
            action_cap=relational_traffic_action_cap,
            reason_cap=relational_traffic_reason_cap,
            reason_traffic_indices=relational_traffic_reason_indices,
        )
        if not self.relational_traffic_enabled:
            for parameter in self.relational_traffic.parameters():
                parameter.requires_grad = False
        self.traffic_adaptive_boundary_enabled = bool(traffic_adaptive_boundary_enabled)
        self.traffic_adaptive_boundary = TIDATrafficAdaptiveBoundary(
            num_actions=num_actions, state_dim=8, cap=traffic_adaptive_boundary_cap
        )
        if not self.traffic_adaptive_boundary_enabled:
            for parameter in self.traffic_adaptive_boundary.parameters():
                parameter.requires_grad = False
        if not self.traffic_trajectory_enabled:
            for parameter in self.traffic_trajectory_head.parameters():
                parameter.requires_grad = False
        self.object_intent_enabled = bool(object_intent_enabled)
        if self.object_intent_enabled:
            self.object_tracker = object_tracker or TIDAFrozenPointTracker.from_local_repository(
                object_tracker_repository,
                model_name=object_tracker_model_name,
                grid_size=object_tracker_grid_size,
            )
        else:
            self.object_tracker = object_tracker
        self.object_intent = TIDAObjectIntentTransport(
            dim=dim,
            num_actions=num_actions,
            num_reasons=num_reasons,
            heads=object_intent_heads,
            action_cap=object_intent_action_cap,
            reason_cap=object_intent_reason_cap,
            reason_traffic_indices=object_intent_reason_indices,
            role_checkpoint=object_intent_role_checkpoint,
        )
        if not self.object_intent_enabled:
            for parameter in self.object_intent.parameters():
                parameter.requires_grad = False
        self.query_identity = nn.Parameter(torch.randn(num_actions + num_predicates, dim) * 0.02)
        self.predicate_identity = nn.Parameter(torch.randn(num_predicates, dim) * 0.02)
        self.num_actions = int(num_actions)
        self.num_predicates = int(num_predicates)
        self.predicate_names = predicate_names

    @staticmethod
    def _swap_lr_name(name: str) -> str:
        return name.replace("left", "__tida_lr__").replace("right", "left").replace("__tida_lr__", "right")

    def _predicate_flip_permutation(self, device: torch.device) -> torch.Tensor:
        lookup = {name: index for index, name in enumerate(self.predicate_names)}
        return torch.tensor(
            [lookup.get(self._swap_lr_name(name), index) for index, name in enumerate(self.predicate_names)],
            device=device,
        )

    def _canonicalize_flipped_image_branch(self, image: dict[str, Any]) -> dict[str, Any]:
        result = dict(image)
        action_perm = torch.tensor([0, 1, 3, 2], device=image["action_logits_final"].device)
        predicate_perm = self._predicate_flip_permutation(image["predicate_tokens"].device)
        result["action_logits_final"] = image["action_logits_final"].index_select(1, action_perm)
        result["action_nodes_primary"] = image["action_nodes_primary"].index_select(1, action_perm)
        result["predicate_tokens"] = image["predicate_tokens"].index_select(1, predicate_perm)
        if "predicate_probs" in image:
            result["predicate_probs"] = image["predicate_probs"].index_select(1, predicate_perm)
        attention = image["predicate_attention"].index_select(1, predicate_perm)
        result["predicate_attention"] = attention.view(
            attention.shape[0], attention.shape[1], 45, 80
        ).flip(3).flatten(2)
        if "ego_features" in image:
            ego = image["ego_features"].view(45, 80, -1).flip(1).clone()
            ego[..., 0] = 1.0 - ego[..., 0]
            ego[..., [3, 4]] = ego[..., [4, 3]]
            result["ego_features"] = ego.flatten(0, 1)
        if "ego_region_masks" in image:
            result["ego_region_masks"] = {
                "front_center": image["ego_region_masks"]["front_center"].view(45, 80).flip(1).flatten(),
                "left_corridor": image["ego_region_masks"]["right_corridor"].view(45, 80).flip(1).flatten(),
                "right_corridor": image["ego_region_masks"]["left_corridor"].view(45, 80).flip(1).flatten(),
                "upper_traffic_region": image["ego_region_masks"]["upper_traffic_region"].view(45, 80).flip(1).flatten(),
                "bottom_drivable_region": image["ego_region_masks"]["bottom_drivable_region"].view(45, 80).flip(1).flatten(),
            }
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.image_model.eval()
        return self

    def owner_parameters(self) -> dict[str, list[nn.Parameter]]:
        owners: dict[str, list[nn.Parameter]] = {}
        for owner, module_name in self.OWNER_MODULES.items():
            owners[owner] = [parameter for parameter in getattr(self, module_name).parameters() if parameter.requires_grad]
        if self.geometric_flow_enabled:
            owners["geometric_action"] = list(self.geometric_heads.action_parameters())
            owners["geometric_reason"] = list(self.geometric_heads.reason_parameters())
        if self.traffic_action_enabled:
            owners["traffic_action"] = [
                parameter for parameter in self.traffic_action.parameters() if parameter.requires_grad
            ]
        if self.traffic_trajectory_enabled:
            utility_ids = {
                id(parameter)
                for module in (
                    self.traffic_trajectory_head.utility_projection,
                    self.traffic_trajectory_head.state_utility_projection,
                )
                for parameter in module.parameters()
            }
            owners["traffic_trajectory"] = [
                parameter for parameter in self.traffic_trajectory_head.parameters()
                if parameter.requires_grad and id(parameter) not in utility_ids
            ]
            owners["traffic_trajectory_utility"] = [
                parameter
                for module in (
                    self.traffic_trajectory_head.utility_projection,
                    self.traffic_trajectory_head.state_utility_projection,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
        if self.relational_traffic_enabled:
            owners["relational_traffic_action"] = [
                parameter
                for module in (
                    self.relational_traffic.action_encoder,
                    self.relational_traffic.action_output,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            owners["relational_traffic_reason"] = [
                parameter
                for module in (
                    self.relational_traffic.reason_encoder,
                    self.relational_traffic.reason_output,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
        if self.traffic_adaptive_boundary_enabled:
            owners["traffic_adaptive_boundary"] = [
                parameter for parameter in self.traffic_adaptive_boundary.parameters()
                if parameter.requires_grad
            ]
        if self.object_intent_enabled:
            owners["object_intent_action"] = [
                parameter
                for module in (
                    self.object_intent.action_encoder,
                    self.object_intent.action_output,
                    self.object_intent.action_pair_encoder,
                    self.object_intent.action_pair_output,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            owners["object_intent_reason"] = [
                parameter
                for module in (
                    self.object_intent.reason_encoder,
                    self.object_intent.reason_output,
                    self.object_intent.reason_pair_encoder,
                    self.object_intent.reason_pair_output,
                )
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            owners["object_intent_action_utility"] = [
                parameter for parameter in self.object_intent.action_utility.parameters()
                if parameter.requires_grad
            ]
            owners["object_intent_reason_utility"] = [
                parameter for parameter in self.object_intent.reason_utility.parameters()
                if parameter.requires_grad
            ]
        # Query identities are the shortcut-free prior for terminal prediction.
        owners["history_reader"] += [self.query_identity, self.predicate_identity]
        return owners

    @staticmethod
    def _target_region_mass(attention: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        masks = TIDATerminalQueryReader._region_masks(grid_hw, attention.device, attention.dtype)
        return torch.einsum("bpn,rn->bpr", attention, masks)

    def decode_encoded_history(
        self,
        history_tokens: torch.Tensor,
        history_region_mass: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        terminal_target_evidence: torch.Tensor,
        terminal_query_identity: torch.Tensor,
        image: dict[str, Any],
        *,
        temporal_action_scale: float | torch.Tensor,
        temporal_reason_scale: float | torch.Tensor,
        geometric: dict[str, torch.Tensor] | None = None,
        history_action_patch_tokens: torch.Tensor | None = None,
        history_action_patch_xy: torch.Tensor | None = None,
        history_action_patch_weight: torch.Tensor | None = None,
        terminal_action_patch_tokens: torch.Tensor | None = None,
        terminal_action_patch_xy: torch.Tensor | None = None,
        terminal_action_patch_weight: torch.Tensor | None = None,
        dense_trajectory_patch_tokens: torch.Tensor | None = None,
        dense_trajectory_grid_hw: tuple[int, int] | None = None,
        history_semantic_patch_tokens: torch.Tensor | None = None,
        history_semantic_patch_xy: torch.Tensor | None = None,
        history_semantic_patch_weight: torch.Tensor | None = None,
        terminal_semantic_patch_tokens: torch.Tensor | None = None,
        terminal_semantic_patch_xy: torch.Tensor | None = None,
        terminal_semantic_patch_weight: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        action_nodes = image["action_nodes_primary"].detach()
        reason_nodes = image["reason_nodes_primary"].detach()
        predicate_tokens = image["predicate_tokens"].detach()
        temporal = self.temporal_encoder(history_tokens, timestamps, frame_valid_mask)
        innovation = self.terminal_innovation(
            terminal_query_identity, temporal["history_summary"], terminal_target_evidence, temporal["history_valid"]
        )
        rho = innovation["innovation_reliability"]
        xi = innovation["innovation_token"]
        target_region_mass = self._target_region_mass(image["predicate_attention"].detach(), (45, 80))
        differential = self.predicate_differential(
            history_tokens[:, :, self.num_actions :], predicate_tokens, xi[:, self.num_actions :],
            timestamps, frame_valid_mask, history_region_mass[:, :, self.num_actions :],
            target_region_mass, rho[:, self.num_actions :],
        )
        # Flow direction must come from ordered history only. Appending the
        # current target after a reversed history creates one artificial final
        # jump that can dominate the EMA and hide the reversal.
        predicate_trajectory = history_tokens[:, :, self.num_actions :]
        predicate_region_trajectory = history_region_mass[:, :, self.num_actions :]
        flow = self.flow_transition_bank(
            predicate_trajectory,
            predicate_region_trajectory,
            timestamps[:, :-1],
            frame_valid_mask[:, :-1],
        )
        image_action = image["action_logits_final"].detach()
        image_reason = image["reason_logits_final"].detach()
        factor_reliability = torch.cat([rho[:, self.num_actions :], rho[:, : self.num_actions]], dim=1)
        action = self.action_reader(
            action_nodes, differential["predicate_differential_state"], xi[:, : self.num_actions],
            factor_reliability, temporal_scale=temporal_action_scale,
            predicate_key_state=differential["predicate_routing_key_state"],
            transition_state=flow["transition_tokens"],
            transition_reliability=flow["transition_reliability"],
            transition_tokens_by_scale=flow["transition_tokens_by_scale"],
            motion_salience=flow["motion_salience"],
            transition_consistency=flow["transition_consistency"],
            history_available=flow["history_available"],
            image_logits=image_action,
        )
        reason = self.reason_reader(
            reason_nodes, differential["predicate_differential_state"],
            action["selected_action_temporal_evidence"], factor_reliability,
            temporal_scale=temporal_reason_scale,
            transition_state=flow["transition_tokens"],
            transition_reliability=flow["transition_reliability"],
            transition_tokens_by_scale=flow["transition_tokens_by_scale"],
            motion_salience=flow["motion_salience"],
            transition_consistency=flow["transition_consistency"],
            history_available=flow["history_available"],
            image_logits=image_reason,
        )
        reason_delta_raw = reason["reason_temporal_delta"]
        geometric = geometric or self._empty_geometric(image_action, image_reason)
        geometric_action_delta_raw = geometric["geometric_action_delta"]
        geometric_reason_delta_raw = geometric["geometric_reason_delta"]
        geometric_prefix_action_delta_raw = geometric["geometric_prefix_action_delta"]
        geometric_prefix_reason_delta_raw = geometric["geometric_prefix_reason_delta"]
        geometric_action_delta = geometric_action_delta_raw * temporal_action_scale
        geometric_reason_delta = geometric_reason_delta_raw * temporal_reason_scale
        geometric_prefix_action_delta = geometric_prefix_action_delta_raw * temporal_action_scale
        geometric_prefix_reason_delta = geometric_prefix_reason_delta_raw * temporal_reason_scale
        traffic = (
            self.traffic_action(
                action_nodes,
                history_tokens[:, :, : self.num_actions],
                timestamps[:, : history_tokens.shape[1]],
                frame_valid_mask[:, : history_tokens.shape[1]],
                image_action,
                patch_tokens=history_action_patch_tokens,
                patch_xy=history_action_patch_xy,
                patch_weight=history_action_patch_weight,
            )
            if self.traffic_action_enabled
            else self._empty_traffic_action(image_action, history_tokens.shape[1])
        )
        traffic_action_delta_raw = traffic["traffic_action_delta"]
        traffic_action_delta = traffic_action_delta_raw * temporal_action_scale
        trajectory = self._empty_traffic_trajectory(image_action, history_tokens.shape[1] + 1)
        if self.traffic_trajectory_enabled:
            required = (
                history_action_patch_tokens, history_action_patch_xy, history_action_patch_weight,
                terminal_action_patch_tokens, terminal_action_patch_xy, terminal_action_patch_weight,
            )
            if any(value is None for value in required):
                raise ValueError("trajectory traffic requires history and terminal action patches")
            trajectory_field = self.traffic_trajectory_builder(
                torch.cat((history_action_patch_tokens, terminal_action_patch_tokens[:, None]), dim=1),
                torch.cat((history_action_patch_xy, terminal_action_patch_xy[:, None]), dim=1),
                torch.cat((history_action_patch_weight, terminal_action_patch_weight[:, None]), dim=1),
                frame_valid_mask,
                dense_patch_tokens=dense_trajectory_patch_tokens,
                dense_grid_hw=dense_trajectory_grid_hw,
            )
            trajectory = {
                **trajectory_field,
                **self.traffic_trajectory_head(
                    action_nodes,
                    trajectory_field["trajectory_appearance"],
                    trajectory_field["trajectory_xy"],
                    trajectory_field["trajectory_visibility"],
                    trajectory_field["trajectory_pair_valid"],
                    trajectory_field["trajectory_common_displacement"],
                    trajectory_field["trajectory_exclusive_displacement"],
                    trajectory_field["trajectory_anchor_weight"],
                    base_action_logits=image_action + action["action_temporal_delta"],
                ),
            }
        traffic_trajectory_delta_raw = trajectory["traffic_trajectory_delta"]
        traffic_trajectory_control_delta_raw = trajectory["traffic_trajectory_control_delta"]
        traffic_trajectory_delta = traffic_trajectory_delta_raw * temporal_action_scale
        traffic_trajectory_control_delta = traffic_trajectory_control_delta_raw * temporal_action_scale
        relational = self._empty_relational_traffic(image_action, image_reason)
        if self.relational_traffic_enabled:
            semantic_required = (
                history_semantic_patch_tokens,
                history_semantic_patch_xy,
                history_semantic_patch_weight,
                terminal_semantic_patch_tokens,
                terminal_semantic_patch_xy,
                terminal_semantic_patch_weight,
            )
            if any(value is None for value in semantic_required):
                raise ValueError("relational traffic requires semantic history and terminal patches")
            semantic_field = self.traffic_trajectory_builder(
                torch.cat(
                    (history_semantic_patch_tokens, terminal_semantic_patch_tokens[:, None]), dim=1
                ),
                torch.cat((history_semantic_patch_xy, terminal_semantic_patch_xy[:, None]), dim=1),
                torch.cat(
                    (history_semantic_patch_weight, terminal_semantic_patch_weight[:, None]), dim=1
                ),
                frame_valid_mask,
                dense_patch_tokens=dense_trajectory_patch_tokens,
                dense_grid_hw=dense_trajectory_grid_hw,
            )
            relational_output = self.relational_traffic(
                    action_nodes,
                    reason_nodes,
                    semantic_field["trajectory_appearance"],
                    semantic_field["trajectory_xy"],
                    semantic_field["trajectory_visibility"],
                    semantic_field["trajectory_pair_valid"],
                    semantic_field["trajectory_common_displacement"],
                    semantic_field["trajectory_exclusive_displacement"],
                    semantic_field["trajectory_anchor_weight"],
                )
            relational = {
                **relational_output,
                **{
                    f"semantic_{key}": value
                    for key, value in semantic_field.items()
                },
            }
        relational_action_delta = relational["relational_action_delta"] * temporal_action_scale
        relational_reason_delta = relational["relational_reason_delta"] * temporal_reason_scale
        semantic_action_delta = action["action_temporal_delta"]
        semantic_reason_delta = reason_delta_raw
        action_delta = (
            semantic_action_delta + geometric_action_delta + traffic_action_delta
            + traffic_trajectory_delta + relational_action_delta
        )
        pre_relational_action_logits = (
            image_action + semantic_action_delta + geometric_action_delta
            + traffic_action_delta + traffic_trajectory_delta
        )
        video_action_base = image_action + action_delta
        if self.traffic_adaptive_boundary_enabled:
            adaptive_boundary = self.traffic_adaptive_boundary(
                video_action_base,
                trajectory["traffic_trajectory_order_delta"],
                trajectory["traffic_trajectory_state_features"],
                trajectory["traffic_trajectory_support"],
                trajectory["trajectory_state_strength"],
                trajectory["trajectory_interaction_risk"],
            )
        else:
            adaptive_boundary = {
                "traffic_adaptive_boundary_delta": torch.zeros_like(video_action_base),
                "traffic_adaptive_deploy_action_logits": video_action_base,
                "traffic_adaptive_boundary_features": video_action_base.new_zeros(
                    video_action_base.shape[0], self.num_actions, 22
                ),
            }
        reason_delta_without_relational = semantic_reason_delta + geometric_reason_delta
        reason_delta_raw = reason_delta_without_relational + relational_reason_delta
        semantic_reason_effective = (
            confidence_aware_reason_delta(
                image_reason, semantic_reason_delta, temperature=self.reason_gate_temperature
            )
            if self.confidence_aware_reason_gate else semantic_reason_delta
        )
        geometric_reason_effective = (
            confidence_aware_reason_delta(
                image_reason, geometric_reason_delta, temperature=self.reason_gate_temperature
            )
            if self.confidence_aware_reason_gate else geometric_reason_delta
        )
        prefix_reason_raw = semantic_reason_delta[:, None] + geometric_prefix_reason_delta
        prefix_reason_effective = (
            confidence_aware_reason_delta(
                image_reason[:, None].expand_as(prefix_reason_raw),
                prefix_reason_raw,
                temperature=self.reason_gate_temperature,
            )
            if self.confidence_aware_reason_gate else prefix_reason_raw
        )
        reason_delta = (
            confidence_aware_reason_delta(
                image_reason, reason_delta_raw, temperature=self.reason_gate_temperature,
            )
            if self.confidence_aware_reason_gate
            else reason_delta_raw
        )
        pre_relational_reason_delta = (
            confidence_aware_reason_delta(
                image_reason,
                reason_delta_without_relational,
                temperature=self.reason_gate_temperature,
            )
            if self.confidence_aware_reason_gate
            else reason_delta_without_relational
        )
        return {
            **temporal, **innovation, **differential, **flow, **action, **reason,
            **geometric, **traffic, **trajectory, **relational, **adaptive_boundary,
            "terminal_target_evidence": terminal_target_evidence,
            "terminal_query_identity": terminal_query_identity,
            "predicate_innovation_token": xi[:, self.num_actions :],
            "predicate_innovation_reliability": rho[:, self.num_actions :],
            "image_action_logits": image_action,
            "semantic_action_temporal_delta": semantic_action_delta,
            "geometric_action_delta": geometric_action_delta,
            "geometric_action_delta_raw": geometric_action_delta_raw,
            "geometric_video_action_logits_raw": image_action + geometric_action_delta_raw,
            "traffic_action_delta_raw": traffic_action_delta_raw,
            "traffic_action_delta": traffic_action_delta,
            "traffic_video_action_logits_raw": image_action + traffic_action_delta_raw,
            "traffic_video_action_logits": image_action + traffic_action_delta,
            "traffic_trajectory_delta_raw": traffic_trajectory_delta_raw,
            "traffic_trajectory_delta": traffic_trajectory_delta,
            "traffic_trajectory_control_delta_raw": traffic_trajectory_control_delta_raw,
            "traffic_trajectory_control_delta": traffic_trajectory_control_delta,
            "relational_action_delta_scaled": relational_action_delta,
            "relational_reason_delta_scaled": relational_reason_delta,
            "pre_relational_video_action_logits": pre_relational_action_logits,
            "pre_relational_video_reason_logits": image_reason + pre_relational_reason_delta,
            "trajectory_video_action_logits_raw": image_action + traffic_trajectory_delta_raw,
            "trajectory_video_action_logits": image_action + traffic_trajectory_delta,
            "semantic_trajectory_video_action_logits_raw": (
                image_action + semantic_action_delta + traffic_trajectory_delta_raw
            ),
            "semantic_trajectory_video_action_logits": (
                image_action + semantic_action_delta + traffic_trajectory_delta
            ),
            "geometric_prefix_action_delta": geometric_prefix_action_delta,
            "geometric_prefix_action_logits_raw": image_action[:, None] + geometric_prefix_action_delta_raw,
            "geometric_prefix_action_logits": image_action[:, None] + geometric_prefix_action_delta,
            "prefix_video_action_logits": image_action[:, None] + semantic_action_delta[:, None] + geometric_prefix_action_delta,
            "action_temporal_delta": action_delta,
            "semantic_video_action_logits": image_action + semantic_action_delta,
            "geometric_video_action_logits": image_action + geometric_action_delta,
            "video_action_logits_base": video_action_base,
            "video_action_logits": adaptive_boundary["traffic_adaptive_deploy_action_logits"],
            "image_reason_logits": image_reason,
            "semantic_reason_temporal_delta": semantic_reason_delta,
            "semantic_reason_temporal_delta_effective": semantic_reason_effective,
            "semantic_video_reason_logits": image_reason + semantic_reason_effective,
            "geometric_reason_delta": geometric_reason_delta,
            "geometric_reason_delta_raw": geometric_reason_delta_raw,
            "geometric_reason_delta_effective": geometric_reason_effective,
            "geometric_video_reason_logits": image_reason + geometric_reason_effective,
            "geometric_prefix_reason_delta": geometric_prefix_reason_delta,
            "geometric_prefix_reason_logits_raw": image_reason[:, None] + geometric_prefix_reason_delta_raw,
            "geometric_prefix_reason_logits": image_reason[:, None] + geometric_prefix_reason_delta,
            "prefix_video_reason_logits": image_reason[:, None] + prefix_reason_effective,
            "reason_temporal_delta_raw": reason_delta_raw,
            "reason_temporal_delta": reason_delta,
            "reason_negative_suppression": (reason_delta_raw - reason_delta).clamp_max(0.0).abs(),
            "video_reason_logits": image_reason + reason_delta,
            "action_temporal_route": action["action_route"],
            "action_null_mass": action["action_route"][..., -1],
            "action_evidence_confidence": action["action_evidence_confidence"],
            "action_effective_trust": action["action_effective_trust"],
            "reason_temporal_route": reason["reason_temporal_attention"],
            "reason_evidence_confidence": reason["reason_evidence_confidence"],
            "reason_effective_trust": reason["reason_effective_trust"],
            "frame_valid_mask": frame_valid_mask,
            "timestamps": timestamps,
            "dynamic_concepts": translate_dynamic_concepts(
                self.predicate_names, differential["predicate_region_mass_velocity"], rho[:, self.num_actions :]
            ),
            "target_predicate_region_mass": target_region_mass,
        }

    def _empty_relational_traffic(
        self, image_action: torch.Tensor, image_reason: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch = image_action.shape[0]
        tracks = self.context_encoder.motion_topk
        dim = self.query_identity.shape[-1]
        return {
            "relational_action_delta": torch.zeros_like(image_action),
            "relational_reason_delta": torch.zeros_like(image_reason),
            "relational_action_candidate": torch.zeros_like(image_action),
            "relational_reason_candidate": torch.zeros_like(image_reason),
            "relational_action_selected_deleted_delta": torch.zeros_like(image_action),
            "relational_action_random_deleted_delta": torch.zeros_like(image_action),
            "relational_reason_selected_deleted_delta": torch.zeros_like(image_reason),
            "relational_reason_random_deleted_delta": torch.zeros_like(image_reason),
            "relational_selected_track": torch.zeros(
                batch, dtype=torch.long, device=image_action.device
            ),
            "relational_random_track": torch.ones(
                batch, dtype=torch.long, device=image_action.device
            ),
            "relational_action_selected_track": torch.zeros(
                batch, self.num_actions, dtype=torch.long, device=image_action.device
            ),
            "relational_action_random_track": torch.ones(
                batch, self.num_actions, dtype=torch.long, device=image_action.device
            ),
            "relational_reason_selected_track": torch.zeros(
                batch, image_reason.shape[1], dtype=torch.long,
                device=image_action.device,
            ),
            "relational_reason_random_track": torch.ones(
                batch, image_reason.shape[1], dtype=torch.long,
                device=image_action.device,
            ),
            "relational_action_attention": image_action.new_zeros(
                batch, self.num_actions, tracks
            ),
            "relational_reason_attention": image_reason.new_zeros(
                batch, image_reason.shape[1], tracks
            ),
            "relational_action_pair_attention": image_action.new_zeros(
                batch, self.num_actions, tracks, tracks
            ),
            "relational_reason_pair_attention": image_reason.new_zeros(
                batch, image_reason.shape[1], tracks, tracks
            ),
            "relational_action_support": torch.zeros_like(image_action),
            "relational_reason_support": torch.zeros_like(image_reason),
            "relational_track_support": image_action.new_zeros(batch, tracks),
            "relational_motion_features": image_action.new_zeros(batch, tracks, 14),
            "relational_pair_features": image_action.new_zeros(batch, tracks, tracks, 8),
            "relational_pair_weights": image_action.new_zeros(batch, tracks, tracks),
            "relational_interaction_risk": image_action.new_zeros(batch, tracks, tracks),
            "semantic_trajectory_xy": image_action.new_zeros(batch, 1, tracks, 1, 2),
        }

    def _empty_traffic_action(
        self, image_action: torch.Tensor, history_frames: int
    ) -> dict[str, torch.Tensor]:
        batch = image_action.shape[0]
        intervals = max(int(history_frames) - 1, 0)
        return {
            "traffic_action_delta": torch.zeros_like(image_action),
            "traffic_action_context": image_action.new_zeros(batch, self.num_actions, self.query_identity.shape[-1]),
            "traffic_action_attention": image_action.new_zeros(
                batch, self.num_actions, intervals * self.num_actions
            ),
            "traffic_same_action_mass": torch.zeros_like(image_action),
            "traffic_motion_energy": image_action.new_zeros(batch, intervals),
            "traffic_history_available": torch.zeros(batch, dtype=torch.bool, device=image_action.device),
            "traffic_patch_displacement": image_action.new_zeros(batch, intervals, self.num_actions, 2),
            "traffic_patch_common_displacement": image_action.new_zeros(batch, intervals, 2),
            "traffic_patch_exclusive_displacement": image_action.new_zeros(
                batch, intervals, self.num_actions, 2
            ),
            "traffic_patch_exclusive_motion_energy": image_action.new_zeros(
                batch, intervals, self.num_actions
            ),
            "traffic_patch_effective_motion": image_action.new_zeros(batch, intervals, self.num_actions),
            "traffic_patch_match_confidence": image_action.new_zeros(batch, intervals, self.num_actions),
            "traffic_patch_motion_energy": image_action.new_zeros(batch, intervals, self.num_actions),
        }

    def _empty_traffic_trajectory(
        self, image_action: torch.Tensor, total_frames: int
    ) -> dict[str, torch.Tensor]:
        batch = image_action.shape[0]
        actions = self.num_actions
        tracks = self.context_encoder.motion_topk
        dim = self.query_identity.shape[-1]
        intervals = max(int(total_frames) - 1, 1)
        return {
            "traffic_trajectory_delta": torch.zeros_like(image_action),
            "traffic_trajectory_control_delta": torch.zeros_like(image_action),
            "traffic_trajectory_credit_logit": torch.zeros_like(image_action),
            "traffic_trajectory_control_logit": torch.zeros_like(image_action),
            "traffic_trajectory_candidate_delta": torch.zeros_like(image_action),
            "traffic_trajectory_order_delta": torch.zeros_like(image_action),
            "traffic_trajectory_state_delta": torch.zeros_like(image_action),
            "traffic_trajectory_state_effective_delta": torch.zeros_like(image_action),
            "traffic_trajectory_state_logit": torch.zeros_like(image_action),
            "traffic_trajectory_state_features": image_action.new_zeros(batch, actions, 8),
            "trajectory_state_strength": torch.zeros_like(image_action),
            "traffic_trajectory_utility_logit": torch.zeros_like(image_action),
            "traffic_trajectory_utility_gate": torch.zeros_like(image_action),
            "traffic_trajectory_order_utility_gate": torch.zeros_like(image_action),
            "traffic_trajectory_state_utility_logit": torch.zeros_like(image_action),
            "traffic_trajectory_state_utility_gate": torch.zeros_like(image_action),
            "traffic_trajectory_context": image_action.new_zeros(batch, actions, dim),
            "traffic_trajectory_trust": image_action.new_zeros(batch, actions),
            "traffic_trajectory_support": image_action.new_zeros(batch, actions),
            "trajectory_support_gate": image_action.new_zeros(batch, actions),
            "trajectory_order_gate": image_action.new_zeros(batch, actions),
            "trajectory_uncertainty_gate": image_action.new_zeros(batch, actions),
            "trajectory_attention": image_action.new_zeros(batch, actions, tracks),
            "trajectory_tokens": image_action.new_zeros(batch, actions, tracks, dim),
            "trajectory_direction_histogram": image_action.new_zeros(batch, actions, tracks, 8),
            "trajectory_speed": image_action.new_zeros(batch, actions, tracks, intervals),
            "trajectory_acceleration": image_action.new_zeros(batch, actions, tracks, intervals),
            "trajectory_radial_motion": image_action.new_zeros(batch, actions, tracks, intervals),
            "trajectory_pair_confidence": image_action.new_zeros(batch, actions, tracks, intervals),
            "trajectory_order_contrast_rms": image_action.new_zeros(batch, actions),
            "trajectory_interaction_risk": image_action.new_zeros(batch, actions, tracks),
            "trajectory_xy": image_action.new_zeros(batch, actions, tracks, total_frames, 2),
            "trajectory_visibility": image_action.new_zeros(batch, actions, tracks, total_frames),
            "trajectory_cycle_confidence": image_action.new_zeros(batch, actions, tracks, total_frames),
            "trajectory_pair_valid": torch.zeros(
                batch, actions, tracks, intervals, dtype=torch.bool, device=image_action.device
            ),
            "trajectory_displacement": image_action.new_zeros(batch, actions, tracks, intervals, 2),
            "trajectory_common_displacement": image_action.new_zeros(batch, intervals, 2),
            "trajectory_exclusive_displacement": image_action.new_zeros(batch, actions, tracks, intervals, 2),
            "trajectory_anchor_weight": image_action.new_zeros(batch, actions, tracks),
            "trajectory_local_candidate_coverage": image_action.new_zeros(
                batch, actions, tracks, total_frames
            ),
        }

    @staticmethod
    def _intervene_patch_history(value: torch.Tensor, intervention: str | None) -> torch.Tensor:
        if intervention == "repeated_last":
            return value[:, -1:].expand_as(value)
        if intervention == "time_reverse":
            return value.flip(1)
        if intervention == "time_shuffle":
            order = torch.cat(
                (torch.arange(0, value.shape[1], 2, device=value.device),
                 torch.arange(1, value.shape[1], 2, device=value.device))
            )
            return value.index_select(1, order)
        return value

    def _empty_geometric(
        self, image_action: torch.Tensor, image_reason: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch = image_action.shape[0]
        zero_action = torch.zeros_like(image_action)
        zero_reason = torch.zeros_like(image_reason)
        return {
            "geometric_action_delta": zero_action,
            "geometric_reason_delta": zero_reason,
            "geometric_prefix_action_delta": zero_action[:, None].expand(-1, 4, -1),
            "geometric_prefix_reason_delta": zero_reason[:, None].expand(-1, 4, -1),
            "geometric_motion_energy": image_action.new_zeros(batch, 1),
            "geometric_global_horizontal": image_action.new_zeros(batch, 1),
            "geometric_global_expansion": image_action.new_zeros(batch, 1),
            "geometric_region_motion": image_action.new_zeros(batch, 1, 5, 3),
            "geometric_flow_field": image_action.new_zeros(batch, 1, 2, 45, 80),
            "geometric_history_available": torch.zeros(batch, dtype=torch.bool, device=image_action.device),
        }

    def _encode_geometric(
        self,
        context_images: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        intervention: str | None = None,
        action_base_logits: torch.Tensor | None = None,
        reason_base_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.geometric_flow_enabled:
            return {}
        frames = context_images
        valid = frame_valid_mask[:, :-1]
        if intervention == "history_off":
            valid = torch.zeros_like(valid)
        elif intervention == "repeated_last":
            frames = frames[:, -1:].expand_as(frames)
        elif intervention == "time_reverse":
            frames = frames.flip(1)
        elif intervention == "time_shuffle":
            order = torch.cat(
                (torch.arange(0, frames.shape[1], 2, device=frames.device),
                 torch.arange(1, frames.shape[1], 2, device=frames.device))
            )
            frames = frames.index_select(1, order)
        measured = self.geometric_flow(frames, valid, timestamps[:, :-1])
        decisions = self.geometric_heads(
            measured["flow_state"], measured["history_available"], action_base_logits, reason_base_logits
        )
        prefixes = self.geometric_heads.forward_prefixes(
            measured["prefix_flow_states"], measured["prefix_available"], action_base_logits, reason_base_logits
        )
        return {
            **decisions,
            **prefixes,
            "geometric_motion_energy": measured["motion_energy"],
            "geometric_global_horizontal": measured["global_horizontal"],
            "geometric_global_expansion": measured["global_expansion"],
            "geometric_region_motion": measured["region_motion"],
            "geometric_flow_field": measured["flow_field"],
            "geometric_history_available": measured["history_available"],
            "geometric_prefix_fractions": measured["prefix_fractions"],
        }

    @staticmethod
    def _intervene_object_tracks(
        xy: torch.Tensor, visibility: torch.Tensor, intervention: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intervention is None:
            return xy, visibility
        history_xy, terminal_xy = xy[:, :-1], xy[:, -1:]
        history_visibility, terminal_visibility = visibility[:, :-1], visibility[:, -1:]
        if intervention == "history_off":
            history_visibility = torch.zeros_like(history_visibility)
        elif intervention == "repeated_last":
            history_xy = terminal_xy.expand_as(history_xy)
            history_visibility = terminal_visibility.expand_as(history_visibility)
        elif intervention == "time_reverse":
            history_xy = history_xy.flip(1)
            history_visibility = history_visibility.flip(1)
        elif intervention == "time_shuffle":
            order = torch.cat(
                (
                    torch.arange(0, history_xy.shape[1], 2, device=xy.device),
                    torch.arange(1, history_xy.shape[1], 2, device=xy.device),
                )
            )
            history_xy = history_xy.index_select(1, order)
            history_visibility = history_visibility.index_select(1, order)
        return (
            torch.cat((history_xy, terminal_xy), dim=1),
            torch.cat((history_visibility, terminal_visibility), dim=1),
        )

    @staticmethod
    def _merge_object_intent(
        temporal_output: dict[str, Any],
        tracker_output: dict[str, torch.Tensor],
        object_output: dict[str, torch.Tensor],
        temporal_action_scale: float | torch.Tensor,
        temporal_reason_scale: float | torch.Tensor,
    ) -> dict[str, Any]:
        action_delta = object_output["object_intent_action_delta"] * temporal_action_scale
        reason_delta = object_output["object_intent_reason_delta"] * temporal_reason_scale
        pre_action = temporal_output["video_action_logits"]
        pre_reason = temporal_output["video_reason_logits"]
        return {
            **temporal_output,
            **tracker_output,
            **object_output,
            "pre_object_intent_video_action_logits": pre_action,
            "pre_object_intent_video_reason_logits": pre_reason,
            "object_intent_action_delta_scaled": action_delta,
            "object_intent_reason_delta_scaled": reason_delta,
            "video_action_logits_base": temporal_output["video_action_logits_base"] + action_delta,
            "video_action_logits": pre_action + action_delta,
            "video_reason_logits": pre_reason + reason_delta,
            "action_temporal_delta": temporal_output["action_temporal_delta"] + action_delta,
            "reason_temporal_delta": temporal_output["reason_temporal_delta"] + reason_delta,
        }

    def rerun_temporal_from_output(
        self,
        output: dict[str, Any],
        intervention: str,
        *,
        temporal_action_scale: float | torch.Tensor,
        temporal_reason_scale: float | torch.Tensor,
    ) -> dict[str, Any]:
        from ..utils.tida_temporal_interventions import apply_query_intervention

        history = apply_query_intervention(
            output["history_query_tokens"], intervention,
            history_valid=output["frame_valid_mask"][:, :-1],
            terminal_predicate_tokens=output["image_branch"]["predicate_tokens"].detach(),
            predicate_indices=output.get("intervention_predicate_indices", ()),
            static_predicate_mask=self.predicate_differential.static_mask,
            action_count=self.num_actions,
        )
        rerun_valid = output["frame_valid_mask"]
        if intervention == "history_off":
            rerun_valid = rerun_valid.clone()
            rerun_valid[:, :-1] = False
        geometric = self._encode_geometric(
            output["_geometric_context_images"], output["timestamps"], output["frame_valid_mask"], intervention,
            output["image_action_logits"], output["image_reason_logits"],
        )
        patch_tokens = self._intervene_patch_history(output["history_action_patch_tokens"], intervention)
        patch_xy = self._intervene_patch_history(output["history_action_patch_xy"], intervention)
        patch_weight = self._intervene_patch_history(output["history_action_patch_weight"], intervention)
        semantic_patch_tokens = self._intervene_patch_history(
            output["history_semantic_patch_tokens"], intervention
        )
        semantic_patch_xy = self._intervene_patch_history(
            output["history_semantic_patch_xy"], intervention
        )
        semantic_patch_weight = self._intervene_patch_history(
            output["history_semantic_patch_weight"], intervention
        )
        dense_history = self._intervene_patch_history(
            output["history_patch_tokens_last"], intervention
        )
        dense_trajectory_patch_tokens = torch.cat(
            (dense_history, output["terminal_patch_tokens_context_grid"][:, None]), dim=1
        )
        rerun = self.decode_encoded_history(
            history, output["history_query_region_mass"], output["timestamps"], rerun_valid,
            output["terminal_target_evidence"], output["terminal_query_identity"], output["image_branch"],
            temporal_action_scale=temporal_action_scale, temporal_reason_scale=temporal_reason_scale,
            geometric=geometric,
            history_action_patch_tokens=patch_tokens,
            history_action_patch_xy=patch_xy,
            history_action_patch_weight=patch_weight,
            terminal_action_patch_tokens=output["terminal_action_patch_tokens"],
            terminal_action_patch_xy=output["terminal_action_patch_xy"],
            terminal_action_patch_weight=output["terminal_action_patch_weight"],
            history_semantic_patch_tokens=semantic_patch_tokens,
            history_semantic_patch_xy=semantic_patch_xy,
            history_semantic_patch_weight=semantic_patch_weight,
            terminal_semantic_patch_tokens=output["terminal_semantic_patch_tokens"],
            terminal_semantic_patch_xy=output["terminal_semantic_patch_xy"],
            terminal_semantic_patch_weight=output["terminal_semantic_patch_weight"],
            dense_trajectory_patch_tokens=dense_trajectory_patch_tokens,
            dense_trajectory_grid_hw=output["history_grid_hw"],
        )
        if not self.object_intent_enabled:
            return rerun
        track_xy, track_visibility = self._intervene_object_tracks(
            output["object_tracks_xy"], output["object_tracks_visibility"], intervention
        )
        object_output = self.object_intent(
            track_xy,
            track_visibility,
            output["_object_terminal_patch_tokens"],
            output["_object_terminal_grid_hw"],
            output["image_branch"]["action_nodes_primary"].detach(),
            output["image_branch"]["reason_nodes_primary"].detach(),
            timestamps=output["timestamps"],
            temporal_patch_tokens=dense_trajectory_patch_tokens,
            temporal_grid_hw=output["history_grid_hw"],
            base_action_logits=rerun["video_action_logits"].detach(),
            base_reason_logits=rerun["video_reason_logits"].detach(),
        )
        return self._merge_object_intent(
            rerun,
            {
                "object_tracks_xy": track_xy,
                "object_tracks_visibility": track_visibility,
                "object_tracks_visibility_rate": track_visibility.float().mean((1, 2)),
            },
            object_output,
            temporal_action_scale,
            temporal_reason_scale,
        )

    def forward(
        self,
        target_image: torch.Tensor,
        context_images: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        *,
        temporal_action_scale: float | torch.Tensor,
        temporal_reason_scale: float | torch.Tensor,
        intervention: str | None = None,
        canonicalize_horizontal_flip: bool = False,
        object_tracks_xy: torch.Tensor | None = None,
        object_tracks_visibility: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            target_field = self.image_model.encode_images(target_image)
            image = self.image_model.decode_from_field(target_field, action_scale=1.0, reason_scale=1.0)
            if canonicalize_horizontal_flip:
                image = self._canonicalize_flipped_image_branch(image)
        action_nodes = image["action_nodes_primary"].detach()
        reason_nodes = image["reason_nodes_primary"].detach()
        predicate_tokens = image["predicate_tokens"].detach()
        target_evidence = torch.cat([action_nodes, predicate_tokens], dim=1)
        terminal_query_identity = self.query_identity[None].expand(target_image.shape[0], -1, -1)
        terminal_read = self.query_reader(
            target_field["patch_tokens_by_layer"], action_nodes, predicate_tokens,
            self.predicate_identity, grid_hw=target_field["grid_hw"],
        )
        terminal_patches = self.context_encoder.select_action_patches(
            target_field, terminal_read["query_attention"][:, : self.num_actions]
        )
        predicate_reliability = image.get(
            "predicate_probs",
            predicate_tokens.new_ones(predicate_tokens.shape[:2]),
        ).detach()
        terminal_semantic_patches = select_semantic_traffic_seeds(
            target_field["patch_tokens_last"],
            terminal_read["query_attention"][:, self.num_actions :],
            predicate_reliability,
            grid_hw=target_field["grid_hw"],
            topk=self.context_encoder.motion_topk,
        )

        context = self.context_encoder(
            context_images, action_nodes, predicate_tokens, self.predicate_identity,
            predicate_reliability=predicate_reliability,
            canonicalize_horizontal_flip=canonicalize_horizontal_flip,
        )
        target_grid_height, target_grid_width = target_field["grid_hw"]
        history_grid_height, history_grid_width = context["history_grid_hw"]
        terminal_patch_tokens_context_grid = F.interpolate(
            target_field["patch_tokens_last"].transpose(1, 2).reshape(
                target_image.shape[0], -1, target_grid_height, target_grid_width
            ),
            size=(history_grid_height, history_grid_width),
            mode="bilinear",
            align_corners=True,
        ).flatten(2).transpose(1, 2)
        history_tokens = context["history_query_tokens"]
        effective_frame_valid_mask = frame_valid_mask
        if intervention is not None:
            from ..utils.tida_temporal_interventions import apply_query_intervention

            history_tokens = apply_query_intervention(
                history_tokens, intervention, history_valid=frame_valid_mask[:, :-1],
                terminal_predicate_tokens=predicate_tokens,
                static_predicate_mask=self.predicate_differential.static_mask,
                action_count=self.num_actions,
            )
            if intervention == "history_off":
                effective_frame_valid_mask = frame_valid_mask.clone()
                effective_frame_valid_mask[:, :-1] = False
        dense_trajectory_patch_tokens = torch.cat(
            (
                self._intervene_patch_history(
                    context["history_patch_tokens_last"], intervention
                ),
                terminal_patch_tokens_context_grid[:, None],
            ),
            dim=1,
        )
        temporal_output = self.decode_encoded_history(
            history_tokens, context["history_query_region_mass"], timestamps, effective_frame_valid_mask,
            target_evidence, terminal_query_identity, image,
            temporal_action_scale=temporal_action_scale,
            temporal_reason_scale=temporal_reason_scale,
            geometric=self._encode_geometric(
                context_images, timestamps, effective_frame_valid_mask, intervention,
                image["action_logits_final"].detach(), image["reason_logits_final"].detach(),
            ),
            history_action_patch_tokens=self._intervene_patch_history(
                context["history_action_patch_tokens"], intervention
            ),
            history_action_patch_xy=self._intervene_patch_history(
                context["history_action_patch_xy"], intervention
            ),
            history_action_patch_weight=self._intervene_patch_history(
                context["history_action_patch_weight"], intervention
            ),
            terminal_action_patch_tokens=terminal_patches["tokens"],
            terminal_action_patch_xy=terminal_patches["xy"],
            terminal_action_patch_weight=terminal_patches["weights"],
            history_semantic_patch_tokens=self._intervene_patch_history(
                context["history_semantic_patch_tokens"], intervention
            ),
            history_semantic_patch_xy=self._intervene_patch_history(
                context["history_semantic_patch_xy"], intervention
            ),
            history_semantic_patch_weight=self._intervene_patch_history(
                context["history_semantic_patch_weight"], intervention
            ),
            terminal_semantic_patch_tokens=terminal_semantic_patches["tokens"],
            terminal_semantic_patch_xy=terminal_semantic_patches["xy"],
            terminal_semantic_patch_weight=terminal_semantic_patches["weights"],
            dense_trajectory_patch_tokens=dense_trajectory_patch_tokens,
            dense_trajectory_grid_hw=context["history_grid_hw"],
        )
        object_output: dict[str, torch.Tensor] = {}
        if self.object_intent_enabled:
            if (object_tracks_xy is None) != (object_tracks_visibility is None):
                raise ValueError("precomputed object tracks require both xy and visibility")
            if object_tracks_xy is None:
                target_for_tracker = F.interpolate(
                    target_image, size=context_images.shape[-2:], mode="bilinear", align_corners=False,
                )
                tracker_context = context_images
                if intervention == "repeated_last":
                    tracker_context = target_for_tracker[:, None].expand_as(context_images)
                elif intervention == "time_reverse":
                    tracker_context = context_images.flip(1)
                elif intervention == "time_shuffle":
                    order = torch.cat((
                        torch.arange(0, context_images.shape[1], 2, device=context_images.device),
                        torch.arange(1, context_images.shape[1], 2, device=context_images.device),
                    ))
                    tracker_context = context_images.index_select(1, order)
                tracker_output = self.object_tracker(
                    torch.cat((tracker_context, target_for_tracker[:, None]), dim=1)
                )
                tracker_xy = tracker_output["object_tracks_xy"]
                tracker_visibility = tracker_output["object_tracks_visibility"]
            else:
                tracker_xy = object_tracks_xy
                tracker_visibility = object_tracks_visibility.bool()
                tracker_output = {
                    "object_tracks_xy": tracker_xy,
                    "object_tracks_visibility": tracker_visibility,
                }
                if intervention is not None:
                    tracker_xy, tracker_visibility = self._intervene_object_tracks(
                        tracker_xy, tracker_visibility, intervention
                    )
            if intervention == "history_off":
                tracker_visibility = tracker_visibility.clone()
                tracker_visibility[:, :-1] = False
            if canonicalize_horizontal_flip:
                tracker_xy = tracker_xy.clone()
                tracker_xy[..., 0] = -tracker_xy[..., 0]
            object_output = self.object_intent(
                tracker_xy,
                tracker_visibility,
                target_field["patch_tokens_last"],
                target_field["grid_hw"],
                action_nodes,
                reason_nodes,
                timestamps=timestamps,
                temporal_patch_tokens=dense_trajectory_patch_tokens,
                temporal_grid_hw=context["history_grid_hw"],
                base_action_logits=temporal_output["video_action_logits"].detach(),
                base_reason_logits=temporal_output["video_reason_logits"].detach(),
            )
            temporal_output = self._merge_object_intent(
                temporal_output,
                {
                    **tracker_output,
                    "object_tracks_xy": tracker_xy,
                    "object_tracks_visibility": tracker_visibility,
                    "object_tracks_visibility_rate": tracker_visibility.float().mean((1, 2)),
                },
                object_output,
                temporal_action_scale,
                temporal_reason_scale,
            )
            temporal_output["_object_terminal_patch_tokens"] = target_field["patch_tokens_last"]
            temporal_output["_object_terminal_grid_hw"] = target_field["grid_hw"]
        return {
            **context,
            **temporal_output,
            "image_branch": image,
            "terminal_action_patch_tokens": terminal_patches["tokens"],
            "terminal_action_patch_xy": terminal_patches["xy"],
            "terminal_action_patch_weight": terminal_patches["weights"],
            "terminal_action_patch_indices": terminal_patches["indices"],
            "terminal_semantic_patch_tokens": terminal_semantic_patches["tokens"],
            "terminal_semantic_patch_xy": terminal_semantic_patches["xy"],
            "terminal_semantic_patch_weight": terminal_semantic_patches["weights"],
            "terminal_semantic_patch_indices": terminal_semantic_patches["indices"],
            "terminal_semantic_predicate_ids": terminal_semantic_patches["predicate_ids"],
            "terminal_patch_tokens_context_grid": terminal_patch_tokens_context_grid,
            "_geometric_context_images": context_images,
        }
