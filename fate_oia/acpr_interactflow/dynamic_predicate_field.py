from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.acpr_scene_predicate_head import ACPRScenePredicateHead
from fate_oia.models.acpr_sparse_ops import entmax15_bisect

from .predicate_ontology import InteractPredicateOntology
from .predicate_transfer import TextPredicateTransfer
from .types import InteractPredicateField


class DynamicPredicateField(nn.Module):
    def __init__(
        self,
        predicate_config: str,
        dim: int = 384,
        num_layers: int = 3,
        source_checkpoint: str | None = None,
        text_encoder_model: str | None = None,
        require_source_checkpoint: bool = False,
        require_transformer_text: bool = False,
    ) -> None:
        super().__init__()
        self.ontology = InteractPredicateOntology(predicate_config)
        self.oia_head = ACPRScenePredicateHead(scene_config="configs/acpr_scene_predicates.yaml", dim=dim, num_layers=num_layers)
        self.psi_queries = nn.Parameter(torch.randn(16, dim) * 0.02)
        self.psi_logit = nn.Linear(dim, 1)
        self.temporal = nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
        self.tcn = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.oia_temporal_logit = nn.Linear(dim, 1)
        self.transfer = TextPredicateTransfer(
            self.ontology.names,
            dim=dim,
            source_checkpoint=source_checkpoint,
            oia_predicate_names=self.ontology.names[:32],
            text_encoder_model=text_encoder_model,
            require_source_checkpoint=require_source_checkpoint,
            require_transformer_text=require_transformer_text,
        )

    @staticmethod
    def _trajectory_indices(anchor_count: int, observed_frames: int, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, anchor_count - 1, observed_frames, device=device).round().long()

    @staticmethod
    def _attention_geometry(attn: torch.Tensor, grid_hw: tuple[int, int] = (45, 80)) -> tuple[torch.Tensor, torch.Tensor]:
        """Return normalized centroids and corridor masses for [B,A,P,N] attention."""
        h, w = grid_hw
        b, a, p, n = attn.shape
        if n != h * w:
            raise ValueError(f"Expected {h*w} attention cells, got {n}")
        maps = attn.reshape(b, a, p, h, w)
        ys = torch.linspace(0.0, 1.0, h, device=attn.device, dtype=attn.dtype).view(1, 1, 1, h, 1)
        xs = torch.linspace(0.0, 1.0, w, device=attn.device, dtype=attn.dtype).view(1, 1, 1, 1, w)
        norm = maps.sum((-2, -1), keepdim=True).clamp_min(1e-9)
        cx = (maps * xs).sum((-2, -1)) / norm.squeeze(-1).squeeze(-1)
        cy = (maps * ys).sum((-2, -1)) / norm.squeeze(-1).squeeze(-1)
        left = maps[..., :, : w // 2].sum((-2, -1))
        right = maps[..., :, w // 2 :].sum((-2, -1))
        front = maps[..., int(h * 0.55) :, int(w * 0.35) : int(w * 0.65)].sum((-2, -1))
        upper = maps[..., : int(h * 0.45), :].sum((-2, -1))
        mass = torch.stack([left, right, front, upper], dim=-1)
        return torch.stack([cx, cy], dim=-1), mass

    @property
    def names(self) -> list[str]:
        return self.ontology.names

    def forward(self, patch_tokens_by_layer: torch.Tensor) -> InteractPredicateField:
        b, a, s, n, d = patch_tokens_by_layer.shape
        flat = patch_tokens_by_layer.reshape(b * a, s, n, d)
        oia = self.oia_head(flat)
        oia_tokens = oia["predicate_tokens"].reshape(b, a, 32, d)
        oia_logits = oia["predicate_logits"].reshape(b, a, 32)
        oia_attn = oia["predicate_attention"].reshape(b, a, 32, n)
        temporal_seq = oia_tokens.transpose(1, 2).reshape(b * 32, a, d)
        temporal_oia, _ = self.temporal(temporal_seq)
        tcn_delta = self.tcn(temporal_oia.transpose(1, 2)).transpose(1, 2)
        temporal_oia_seq = temporal_oia + 0.25 * tcn_delta
        temporal_oia = temporal_oia_seq[:, -1].reshape(b, 32, d)
        temporal_oia_anchor = temporal_oia_seq.reshape(b, 32, a, d).transpose(1, 2)
        oia_logits = oia_logits + 0.10 * self.oia_temporal_logit(temporal_oia_anchor).squeeze(-1)
        oia_logits_t = oia_logits.mean(1)
        psi_base_anchor = patch_tokens_by_layer.mean(2)
        psi_score_anchor = torch.einsum("pd,band->bapn", self.psi_queries, psi_base_anchor) / (d ** 0.5)
        psi_attn_anchor = entmax15_bisect(psi_score_anchor, dim=-1)
        psi_tokens_anchor = torch.einsum("bapn,band->bapd", psi_attn_anchor, psi_base_anchor)
        psi_logits_anchor = self.psi_logit(psi_tokens_anchor).squeeze(-1)
        psi_tokens = psi_tokens_anchor.mean(1)
        psi_logits = psi_logits_anchor.mean(1)
        tokens = torch.cat([temporal_oia, psi_tokens], dim=1)
        logits = torch.cat([oia_logits_t, psi_logits], dim=1)
        probs = torch.sigmoid(logits)
        attention_anchor = torch.cat([oia_attn, psi_attn_anchor], dim=2)
        attention = attention_anchor.mean(1)
        transfer = self.transfer(tokens)
        transfer_gate = transfer["transfer_gate"].to(tokens.device, tokens.dtype).view(1, -1, 1)
        tokens = tokens + 0.1 * transfer_gate * transfer["transferred_predicate_tokens"]
        traj_index = self._trajectory_indices(a, 15, patch_tokens_by_layer.device)
        logits_anchor = torch.cat([oia_logits, psi_logits_anchor], dim=2)
        probs_anchor = torch.sigmoid(logits_anchor)
        tokens_anchor = torch.cat([oia_tokens, psi_tokens_anchor], dim=2)
        evidence_maps_anchor = attention_anchor.reshape(b, a, 48, 45, 80)
        centroids_anchor, corridor_mass_anchor = self._attention_geometry(attention_anchor)
        predicate_logits_trajectory = logits_anchor[:, traj_index]
        predicate_probs_trajectory = probs_anchor[:, traj_index]
        predicate_token_trajectory = tokens_anchor[:, traj_index]
        predicate_evidence_maps = evidence_maps_anchor[:, traj_index]
        predicate_centroids = centroids_anchor[:, traj_index]
        predicate_corridor_mass = corridor_mass_anchor[:, traj_index]
        predicate_confidence = (predicate_probs_trajectory - 0.5).abs() * 2.0
        temporal_stats = {
            "predicate_count": 48,
            "temporal_anchor_count": a,
            "predicate_trajectory_length": int(predicate_logits_trajectory.shape[1]),
            "predicate_positive_rate": float((probs > 0.5).float().mean().detach().cpu()),
            "attention_entropy": float((-(attention.clamp_min(1e-9).log() * attention).sum(-1)).mean().detach().cpu()),
            "transfer_gate_mean": float(transfer["transfer_gate"].detach().mean().cpu()),
            "text_embedding_source": transfer.get("text_embedding_source", "unknown"),
            "oia_source_loaded": bool(self.transfer.report().get("source_loaded", False)),
        }
        return InteractPredicateField(
            predicate_logits=logits,
            predicate_probs=probs,
            predicate_logits_trajectory=predicate_logits_trajectory,
            predicate_probs_trajectory=predicate_probs_trajectory,
            predicate_tokens=tokens,
            predicate_token_trajectory=predicate_token_trajectory,
            predicate_attention=attention,
            predicate_evidence_maps=predicate_evidence_maps,
            predicate_confidence=predicate_confidence,
            predicate_centroids=predicate_centroids,
            predicate_corridor_mass=predicate_corridor_mass,
            transfer_gate=transfer["transfer_gate"],
            predicate_names=self.ontology.names,
            temporal_stats=temporal_stats,
        )
