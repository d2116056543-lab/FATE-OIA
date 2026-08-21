from __future__ import annotations

import math

import torch
from torch import nn


class TIDAContinuousTimeEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(0.25), math.log(16.0), max(dim // 2, 1)))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.proj = nn.Linear(frequencies.numel() * 2, dim)

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        phase = timestamps[..., None] * self.frequencies
        return self.proj(torch.cat([phase.sin(), phase.cos()], dim=-1))


class TIDATemporalEncoder(nn.Module):
    def __init__(self, dim: int = 384, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.time_encoding = TIDAContinuousTimeEncoding(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(dim))

    def forward(
        self, history_tokens: torch.Tensor, timestamps: torch.Tensor, frame_valid_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if history_tokens.ndim != 4:
            raise ValueError("history_tokens must be [B,T,Q,D]")
        batch, frames, queries, dim = history_tokens.shape
        if timestamps.shape != (batch, frames + 1) or frame_valid_mask.shape != (batch, frames + 1):
            raise ValueError("timestamps/frame_valid_mask must include history plus target")
        history_valid = frame_valid_mask[:, :frames]
        encoded_input = history_tokens + self.time_encoding(timestamps[:, :frames])[:, :, None]
        flat = encoded_input.transpose(1, 2).reshape(batch * queries, frames, dim)
        valid_flat = history_valid[:, None].expand(-1, queries, -1).reshape(batch * queries, frames)
        # Transformer disallows a row whose every key is masked; unmask one zeroed slot then erase the result.
        all_invalid = ~valid_flat.any(dim=1)
        safe_valid = valid_flat.clone()
        safe_valid[all_invalid, 0] = True
        flat = flat.clone()
        flat[all_invalid, 0] = 0
        causal_mask = torch.triu(torch.ones(frames, frames, device=flat.device, dtype=torch.bool), diagonal=1)
        states = self.encoder(flat, mask=causal_mask, src_key_padding_mask=~safe_valid)
        states[all_invalid] = 0
        states = states.reshape(batch, queries, frames, dim)
        valid_count = history_valid.sum(-1)
        last_valid = (history_valid.to(torch.long).cumsum(-1) * history_valid).argmax(-1)
        gather_index = last_valid[:, None, None, None].expand(-1, queries, 1, dim)
        summary = states.gather(2, gather_index).squeeze(2)
        has_history = valid_count > 0
        summary = summary * has_history[:, None, None].to(summary.dtype)
        return {
            "history_states": states,
            "history_summary": summary,
            "history_valid": has_history,
            "causal_mask": causal_mask,
        }
