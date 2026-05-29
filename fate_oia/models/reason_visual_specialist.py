from __future__ import annotations

import torch
from torch import nn


class ReasonVisualSpecialist(nn.Module):
    """Visual-aware delta head for BDD-OIA reason logits.

    The module is intentionally additive: it predicts a bounded delta that is
    added to Run C reason logits. This keeps initialization near Run C while
    still giving non-zero gradients through visual/label tokens.
    """

    def __init__(
        self,
        dim: int = 384,
        reason_dim: int = 21,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        ffn_dim: int | None = None,
        max_delta_scale: float = 0.25,
        init_delta_scale: float = 0.05,
        topk_visual_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.max_delta_scale = float(max_delta_scale)
        self.topk_visual_tokens = topk_visual_tokens
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim or dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(dim)
        self.delta_head = nn.Linear(dim, 1)
        nn.init.normal_(self.delta_head.weight, std=0.01)
        nn.init.zeros_(self.delta_head.bias)
        ratio = min(0.99, max(1e-4, float(init_delta_scale) / max(self.max_delta_scale, 1e-6)))
        self.delta_scale_raw = nn.Parameter(torch.full((reason_dim,), torch.logit(torch.tensor(ratio)).item()))

    def _select_visual_tokens(self, visual_tokens: torch.Tensor, attention: torch.Tensor | None) -> torch.Tensor:
        if self.topk_visual_tokens is None or visual_tokens.shape[1] <= self.topk_visual_tokens:
            return visual_tokens
        if attention is None:
            scores = visual_tokens.norm(dim=-1)
        else:
            attn = attention
            if attn.ndim == 4:
                attn = attn.mean(1)
            if attn.ndim == 3:
                # [B, labels, N] -> reason-label mean over tokens.
                scores = attn[:, -self.reason_dim :, : visual_tokens.shape[1]].mean(1)
            else:
                scores = visual_tokens.norm(dim=-1)
        keep = min(int(self.topk_visual_tokens), visual_tokens.shape[1])
        idx = torch.sort(torch.topk(scores, k=keep, dim=1).indices, dim=1).values
        return torch.gather(visual_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, visual_tokens.shape[-1]))

    def effective_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.delta_scale_raw) * self.max_delta_scale

    def forward(
        self,
        visual_tokens: torch.Tensor,
        label_tokens: torch.Tensor,
        base_reason_logits: torch.Tensor | None = None,
        attention: torch.Tensor | None = None,
        evidence_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | list[float]]]:
        if visual_tokens.ndim != 3 or label_tokens.ndim != 3:
            raise ValueError("visual_tokens and label_tokens must be [B,N,D]")
        bsz, _, dim = visual_tokens.shape
        visual = self._select_visual_tokens(visual_tokens, attention)
        memory_parts = [visual, label_tokens]
        if evidence_tokens is not None and evidence_tokens.numel() > 0:
            memory_parts.append(evidence_tokens)
        memory = torch.cat(memory_parts, dim=1)
        queries = self.reason_queries.unsqueeze(0).expand(bsz, -1, -1)
        decoded = self.decoder(queries, memory)
        decoded = self.norm(decoded)
        raw_delta = self.delta_head(decoded).squeeze(-1)
        scale = self.effective_scale().to(dtype=raw_delta.dtype, device=raw_delta.device)
        delta = raw_delta * scale.view(1, -1)
        diagnostics = {
            "reason_delta_abs_mean": float(delta.detach().abs().mean().item()),
            "reason_delta_abs_max": float(delta.detach().abs().max().item()),
            "reason_delta_per_label_mean": delta.detach().mean(0).cpu().tolist(),
            "reason_specialist_scale_mean": float(scale.detach().mean().item()),
            "reason_specialist_scale_per_label": scale.detach().cpu().tolist(),
        }
        return delta, diagnostics
