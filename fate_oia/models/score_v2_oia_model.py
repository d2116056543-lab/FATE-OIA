from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fate_oia.models.adaptformer_lite import AdaptFormerLite
from fate_oia.models.multilayer_dino_features import MultiLayerDINOFeatureFusion
from fate_oia.models.semantic_label_queries import SemanticLabelQueries
from fate_oia.models.strong_label_decoder import StrongLabelDecoder


@dataclass
class ScoreV2OIAConfig:
    dim: int = 384
    action_dim: int = 4
    reason_dim: int = 21
    n_last_blocks: int = 4
    num_heads: int = 6
    decoder_self_layers: int = 2
    dropout: float = 0.1
    use_adaptformer: bool = False
    adaptformer_bottleneck_dim: int = 64


class ScoreV2OIAModel(nn.Module):
    """Score-focused BDD-OIA branch using DINO multilayer patch tokens."""

    def __init__(self, config: ScoreV2OIAConfig) -> None:
        super().__init__()
        self.config = config
        self.fusion = MultiLayerDINOFeatureFusion(config.dim, config.n_last_blocks, dropout=config.dropout)
        self.queries = SemanticLabelQueries(
            num_labels=config.action_dim + config.reason_dim,
            dim=config.dim,
            action_dim=config.action_dim,
            dropout=config.dropout,
        )
        self.adapter = (
            AdaptFormerLite(config.dim, bottleneck_dim=config.adaptformer_bottleneck_dim, dropout=config.dropout)
            if config.use_adaptformer
            else nn.Identity()
        )
        self.decoder = StrongLabelDecoder(
            dim=config.dim,
            action_dim=config.action_dim,
            reason_dim=config.reason_dim,
            num_heads=config.num_heads,
            self_layers=config.decoder_self_layers,
            dropout=config.dropout,
        )

    def forward(self, dino_layers: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor | str]:
        fused = self.fusion(dino_layers)
        tokens = self.adapter(fused["tokens"])
        label_queries = self.queries(tokens.shape[0], device=tokens.device)
        decoded = self.decoder(label_queries, tokens)
        decoded["layer_weights"] = fused["layer_weights"]
        decoded["tokens"] = tokens
        decoded["score_v2_stage"] = "score_v2_adaptformer" if self.config.use_adaptformer else "score_v2_patch_only"
        return decoded
