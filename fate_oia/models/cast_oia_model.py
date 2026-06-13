from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml

from fate_oia.models.cast_action_set_energy import CastActionSetEnergy
from fate_oia.models.cast_dino_field import CastDinoFieldExtractor
from fate_oia.models.cast_ego_encoding import EgoPatchCoordinateEncoder
from fate_oia.models.cast_evidence_graph import CastEvidenceGraph
from fate_oia.models.cast_label_evidence import CastLabelEvidence
from fate_oia.models.cast_reason_reliability import CastReasonReliability
from fate_oia.models.cast_text_encoder import CastLabelQueryBuilder, build_label_texts


class CastOIAModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        ontology_path: str | Path = "configs/cast_oia_label_ontology.yaml",
        use_dino: bool = True,
        grid_hw: tuple[int, int] = (45, 80),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        checkpoint_key: str = "teacher",
        selected_layers: tuple[int, ...] = (3, 7, 11),
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.use_dino = bool(use_dino)
        self.grid_hw = tuple(grid_hw)
        self.selected_layers = tuple(selected_layers)
        if self.use_dino:
            self.dino = CastDinoFieldExtractor(
                arch="vit_small",
                patch_size=8,
                pretrained_weights=str(pretrained_weights),
                checkpoint_key=checkpoint_key,
                selected_layers=selected_layers,
                freeze_backbone=True,
            )
            dino_dim = self.dino.embed_dim
            self.input_proj = nn.Identity() if dino_dim == dim else nn.Linear(dino_dim, dim)
        else:
            self.dino = None
            self.synthetic_patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
            self.input_proj = nn.Identity()
        ontology = yaml.safe_load(Path(ontology_path).read_text(encoding="utf-8"))
        self.label_texts = build_label_texts(ontology)
        self.query_builder = CastLabelQueryBuilder(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.ego = EgoPatchCoordinateEncoder(dim=dim, grid_hw=self.grid_hw)
        self.label_evidence = CastLabelEvidence(dim=dim, num_labels=action_dim + reason_dim, selected_layers=len(selected_layers))
        self.set_node_seed = nn.Parameter(torch.zeros(16, dim))
        self.graph = CastEvidenceGraph(dim=dim, num_labels=action_dim + reason_dim, num_sets=16, topk_edges=16)
        self.action_set = CastActionSetEnergy(dim=dim, action_dim=action_dim)
        self.base_action_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.action_fusion_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.reason = CastReasonReliability(dim=dim, reason_dim=reason_dim)
        nn.init.normal_(self.set_node_seed, std=0.02)
        nn.init.constant_(self.action_fusion_gate[-1].bias, -4.0)

    def _extract_field(self, images: torch.Tensor) -> dict[str, Any]:
        if self.use_dino:
            with torch.no_grad():
                field = self.dino(images)
            patches = self.input_proj(field["patch_tokens_by_layer"])
            return {
                "patch_tokens_by_layer": patches,
                "grid_hw": field["grid_hw"],
                "original_tokens": field["original_tokens"],
            }
        patches = self.synthetic_patch(images).flatten(2).transpose(1, 2)
        h, w = images.shape[-2] // 8, images.shape[-1] // 8
        self.ego.grid_hw = (h, w)
        return {
            "patch_tokens_by_layer": torch.stack([patches, patches * 0.7, patches * 1.3], dim=1),
            "grid_hw": (h, w),
            "original_tokens": patches.shape[1] + 1,
        }

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        field = self._extract_field(images)
        patch_tokens = field["patch_tokens_by_layer"]
        self.ego.grid_hw = tuple(field["grid_hw"])
        patch_tokens, ego_features = self.ego(patch_tokens)
        q = self.query_builder(self.label_texts)
        label_queries = q["label_queries"].to(images.device, images.dtype)
        text_similarity = q["text_similarity_matrix"].to(images.device, images.dtype)
        ev = self.label_evidence(label_queries, patch_tokens, ego_features)
        label_nodes = label_queries.view(1, -1, self.dim).expand(images.shape[0], -1, -1) + ev["label_evidence"]
        base_action_logits = self.base_action_head(label_nodes[:, : self.action_dim]).squeeze(-1)
        set_nodes = self.set_node_seed.view(1, 16, self.dim).expand(images.shape[0], -1, -1)
        graph = self.graph(label_nodes, ev["label_evidence"], ev["label_attention"], set_nodes, text_similarity)
        updated_labels = graph["updated_label_nodes"]
        updated_sets = graph["updated_set_nodes"]
        graph_context = updated_sets.mean(1)
        aset = self.action_set(updated_labels[:, : self.action_dim], graph_context, updated_sets)
        cast_action_logits = aset["action_logits"]
        action_fusion_gate = torch.sigmoid(self.action_fusion_gate(updated_labels[:, : self.action_dim]).squeeze(-1))
        bounded_action_delta = 2.0 * torch.tanh((cast_action_logits - base_action_logits) / 2.0)
        action_logits = base_action_logits + action_fusion_gate * bounded_action_delta
        reason_nodes = updated_labels[:, self.action_dim :]
        graph_support = torch.sigmoid(graph["reason_to_set_logits"]).mean(-1)
        evidence_conf = ev["label_attention"][:, self.action_dim :].amax(-1)
        logit_margin = graph_support - graph_support.mean(-1, keepdim=True)
        reason = self.reason(reason_nodes, ev["label_evidence"][:, self.action_dim :], graph_support, evidence_conf, logit_margin)
        evidence_stats = dict(ev["attention_stats"])
        evidence_stats["original_tokens"] = int(field["original_tokens"])
        evidence_stats["grid_hw"] = list(field["grid_hw"])
        return {
            "action_logits": action_logits,
            "base_action_logits": base_action_logits,
            "cast_action_logits": cast_action_logits,
            "action_fusion_gate": action_fusion_gate,
            "bounded_action_delta": bounded_action_delta,
            "reason_logits": reason["reason_logits"],
            "action_set_logits": aset["action_set_logits"],
            "action_set_probs": aset["action_set_probs"],
            "action_marginal_probs": aset["action_marginal_probs"],
            "atomic_action_logits": aset["atomic_logits"],
            "pair_logits": aset["pair_logits"],
            "cardinality_logits": aset["cardinality_logits"],
            "label_attention": ev["label_attention"],
            "label_evidence": ev["label_evidence"],
            "label_layer_weights": ev["label_layer_weights"],
            "graph_edge_weights": graph["edge_weights"],
            "reason_to_set_logits": graph["reason_to_set_logits"],
            "reason_reliability": reason["reason_reliability"],
            "evidence_stats": evidence_stats,
            "graph_stats": graph["graph_stats"],
            "action_set_stats": {},
            "text_similarity_matrix": text_similarity,
        }
