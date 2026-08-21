from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .tida_action_reader import TIDAActionReader
from .tida_context_encoder import TIDAContextEncoder
from .tida_predicate_differential import TIDAPredicateDifferential
from .tida_reason_reader import TIDAReasonReader
from .tida_temporal_encoder import TIDATemporalEncoder
from .tida_terminal_innovation import TIDATerminalInnovation
from .tida_terminal_query_reader import TIDATerminalQueryReader
from ..explain.tida_dynamic_concepts import translate_dynamic_concepts


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
            self.image_model.foundation.dino, self.query_reader, context_chunk_size=context_chunk_size
        )
        self.temporal_encoder = TIDATemporalEncoder(dim=dim, num_layers=2, num_heads=4, dropout=0.10)
        self.terminal_innovation = TIDATerminalInnovation(dim=dim)
        self.predicate_differential = TIDAPredicateDifferential(
            dim=dim,
            predicate_names=predicate_names,
            roles=predicate_roles,
            role_path=predicate_role_path,
        )
        self.action_reader = TIDAActionReader(dim, num_actions, num_predicates, kappa=0.15)
        self.reason_reader = TIDAReasonReader(dim, num_reasons, kappa=0.12)
        self.query_identity = nn.Parameter(torch.randn(num_actions + num_predicates, dim) * 0.02)
        self.predicate_identity = nn.Parameter(torch.randn(num_predicates, dim) * 0.02)
        self.static_context_projection = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim))
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
        # Query identities and static context are part of target-conditioned history reading.
        owners["history_reader"] += [self.query_identity, self.predicate_identity]
        owners["history_reader"] += list(self.static_context_projection.parameters())
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
        terminal_static_context: torch.Tensor,
        image: dict[str, Any],
        *,
        temporal_action_scale: float | torch.Tensor,
        temporal_reason_scale: float | torch.Tensor,
    ) -> dict[str, Any]:
        action_nodes = image["action_nodes_primary"].detach()
        reason_nodes = image["reason_nodes_primary"].detach()
        predicate_tokens = image["predicate_tokens"].detach()
        temporal = self.temporal_encoder(history_tokens, timestamps, frame_valid_mask)
        innovation = self.terminal_innovation(
            terminal_static_context, temporal["history_summary"], terminal_target_evidence, temporal["history_valid"]
        )
        rho = innovation["innovation_reliability"]
        xi = innovation["innovation_token"]
        target_region_mass = self._target_region_mass(image["predicate_attention"].detach(), (45, 80))
        differential = self.predicate_differential(
            history_tokens[:, :, self.num_actions :], predicate_tokens, xi[:, self.num_actions :],
            timestamps, frame_valid_mask, history_region_mass[:, :, self.num_actions :],
            target_region_mass, rho[:, self.num_actions :],
        )
        factor_reliability = torch.cat([rho[:, self.num_actions :], rho[:, : self.num_actions]], dim=1)
        action = self.action_reader(
            action_nodes, differential["predicate_differential_state"], xi[:, : self.num_actions],
            factor_reliability, temporal_scale=temporal_action_scale,
            predicate_key_state=differential["predicate_routing_key_state"],
        )
        reason = self.reason_reader(
            reason_nodes, differential["predicate_differential_state"],
            action["selected_action_temporal_evidence"], factor_reliability,
            temporal_scale=temporal_reason_scale,
        )
        image_action = image["action_logits_final"].detach()
        image_reason = image["reason_logits_final"].detach()
        return {
            **temporal, **innovation, **differential, **action, **reason,
            "terminal_target_evidence": terminal_target_evidence,
            "terminal_static_context": terminal_static_context,
            "image_action_logits": image_action,
            "video_action_logits": image_action + action["action_temporal_delta"],
            "image_reason_logits": image_reason,
            "video_reason_logits": image_reason + reason["reason_temporal_delta"],
            "action_temporal_route": action["action_route"],
            "action_null_mass": action["action_route"][..., -1],
            "reason_temporal_route": reason["reason_temporal_attention"],
            "frame_valid_mask": frame_valid_mask,
            "timestamps": timestamps,
            "dynamic_concepts": translate_dynamic_concepts(
                self.predicate_names, differential["predicate_region_mass_velocity"], rho[:, self.num_actions :]
            ),
            "target_predicate_region_mass": target_region_mass,
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
        return self.decode_encoded_history(
            history, output["history_query_region_mass"], output["timestamps"], rerun_valid,
            output["terminal_target_evidence"], output["terminal_static_context"], output["image_branch"],
            temporal_action_scale=temporal_action_scale, temporal_reason_scale=temporal_reason_scale,
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
        global_token = image["cls_tokens_by_layer"].detach().mean(1)
        static_token = predicate_tokens[:, self.predicate_differential.static_mask].mean(1)
        static_context = self.static_context_projection(torch.cat([global_token, static_token], dim=-1))[:, None]
        static_context = static_context + self.query_identity[None]

        context = self.context_encoder(
            context_images, action_nodes, predicate_tokens, self.predicate_identity,
            canonicalize_horizontal_flip=canonicalize_horizontal_flip,
        )
        history_tokens = context["history_query_tokens"]
        if intervention is not None:
            from ..utils.tida_temporal_interventions import apply_query_intervention

            history_tokens = apply_query_intervention(
                history_tokens, intervention, history_valid=frame_valid_mask[:, :-1],
                terminal_predicate_tokens=predicate_tokens,
                static_predicate_mask=self.predicate_differential.static_mask,
                action_count=self.num_actions,
            )
        temporal_output = self.decode_encoded_history(
            history_tokens, context["history_query_region_mass"], timestamps, frame_valid_mask,
            target_evidence, static_context, image,
            temporal_action_scale=temporal_action_scale,
            temporal_reason_scale=temporal_reason_scale,
        )
        return {
            **context,
            **temporal_output,
            "image_branch": image,
        }
