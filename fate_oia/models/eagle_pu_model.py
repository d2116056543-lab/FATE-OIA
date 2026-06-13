from __future__ import annotations

import torch
from torch import nn

from .eagle_pu_action_set_aux import ActionSetAuxiliaryHead
from .eagle_pu_calibration import EaglePUCalibrationHead
from .eagle_pu_dino_field import EaglePUDinoFieldExtractor
from .eagle_pu_ego_encoding import EaglePUEgoEncoding
from .eagle_pu_label_trunk import EaglePULabelDecisionTrunk
from .eagle_pu_proto_transport import ReasonPrototypeTransport
from .eagle_pu_reason_reliability import EaglePUReasonReliability
from .eagle_pu_state_bank import ObjectiveEnvironmentStateBank
from .eagle_pu_state_graph import StateGroundedLabelGraph
from .eagle_pu_text_encoder import HashingTextPrototypeEncoder


class EaglePUModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        dino_dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        ontology_path: str = "configs/eagle_pu_reason_ontology.yaml",
        freeze_dino: bool = True,
        use_mock_dino: bool = False,
        use_action_graph_delta: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.dino = EaglePUDinoFieldExtractor(selected_layers=selected_layers, pretrained_weights=pretrained_weights, freeze_backbone=freeze_dino, use_mock_dino=use_mock_dino, mock_dim=dino_dim)
        self.input_proj = nn.Linear(dino_dim, dim) if dino_dim != dim else nn.Identity()
        self.ego = EaglePUEgoEncoding(grid_hw=(45, 80), dim=dim)
        self.text_encoder = HashingTextPrototypeEncoder(ontology_path=ontology_path, dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.state_bank = ObjectiveEnvironmentStateBank(dim=dim, num_layers=len(selected_layers), num_states=24)
        self.trunk = EaglePULabelDecisionTrunk(dim=dim, action_dim=action_dim, reason_dim=reason_dim, num_layers=len(selected_layers))
        self.proto_transport = ReasonPrototypeTransport(dim=dim, reason_dim=reason_dim, num_prototypes=6)
        self.state_graph = StateGroundedLabelGraph(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_action_delta=use_action_graph_delta)
        self.action_set_aux = ActionSetAuxiliaryHead(dim=dim, action_dim=action_dim)
        self.reason_reliability_head = EaglePUReasonReliability(dim=dim, reason_dim=reason_dim)
        self.calibration = EaglePUCalibrationHead(num_labels=self.num_labels)
        self.proto_gate_head = nn.Linear(dim, reason_dim)
        self.graph_gate_head = nn.Linear(dim, reason_dim)

    def forward(
        self,
        images: torch.Tensor,
        epoch: int = 0,
        patch_delete_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict | tuple[int, int] | int]:
        field = self.dino(images)
        patch = self.input_proj(field["patch_tokens_by_layer"])
        if patch_delete_mask is not None:
            mask = patch_delete_mask.to(device=patch.device, dtype=patch.dtype)
            if mask.ndim != 2 or mask.shape[-1] != patch.shape[2]:
                raise ValueError(f"patch_delete_mask must be [B,{patch.shape[2]}], got {tuple(mask.shape)}")
            patch = patch * (1.0 - mask[:, None, :, None])
        cls = self.input_proj(field["cls_tokens_by_layer"])
        # Freeze DINO features but keep downstream modules trainable.
        ego_tokens, ego_stats = self.ego(patch[:, 0])
        text = self.text_encoder()
        state = self.state_bank(patch, ego_tokens=ego_tokens)
        trunk = self.trunk(patch, text["label_queries"], state_tokens=state["state_tokens"])
        label_nodes = trunk["label_nodes"]
        reason_nodes = label_nodes[:, self.action_dim :]
        proto = self.proto_transport(reason_nodes, state["state_tokens"], epoch=epoch)
        graph = self.state_graph(label_nodes, state["state_tokens"], text_similarity=text["text_similarity_matrix"])
        action_set = self.action_set_aux(label_nodes, trunk["action_logits_direct"])
        graph_support = torch.sigmoid(graph["reason_to_set_logits"].amax(-1))
        evidence_confidence = trunk["label_attention"][:, self.action_dim:].amax(-1)
        logit_margin = trunk["reason_logits_direct"].abs().detach()
        reliability = self.reason_reliability_head(reason_nodes, reason_nodes, graph_support, evidence_confidence, logit_margin)
        state_summary = state["state_tokens"].mean(1)
        proto_gate = torch.sigmoid(self.proto_gate_head(state_summary))
        graph_gate = torch.sigmoid(self.graph_gate_head(state_summary))
        reason_logits_direct_plus_prototype = trunk["reason_logits_direct"] + proto_gate * proto["prototype_reason_delta"]
        reason_logits_direct_plus_graph = trunk["reason_logits_direct"] + graph_gate * graph["reason_graph_delta"]
        reason_logits_final_raw = reason_logits_direct_plus_prototype + graph_gate * graph["reason_graph_delta"]
        action_logits_final_raw = trunk["action_logits_direct"]
        raw_all = torch.cat([action_logits_final_raw, reason_logits_final_raw], dim=-1)
        calibrated_all, calib_temperature, calib_bias = self.calibration(raw_all)
        out = {
            **field,
            **text,
            **state,
            **trunk,
            **proto,
            **graph,
            **action_set,
            **reliability,
            "state_group_logits": state["state_group_logits"],
            "state_layer_weights": state["state_layer_weights"],
            "ego_stats": ego_stats,
            "cls_tokens_by_layer_projected": cls,
            "proto_gate": proto_gate,
            "graph_gate": graph_gate,
            "reason_logits_direct_plus_prototype": reason_logits_direct_plus_prototype,
            "reason_logits_direct_plus_graph": reason_logits_direct_plus_graph,
            "action_logits_final_raw": action_logits_final_raw,
            "reason_logits_final_raw": reason_logits_final_raw,
            "action_logits_final_calibrated": calibrated_all[:, : self.action_dim],
            "reason_logits_final_calibrated": calibrated_all[:, self.action_dim :],
            "calibration_temperature": calib_temperature,
            "calibration_bias": calib_bias,
        }
        return out



