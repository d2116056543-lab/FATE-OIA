from __future__ import annotations

import torch
from torch import nn


class _LabelCorrelationLayer(nn.Module):
    """Transformer-style label self-attention layer with an optional LxL prior bias."""

    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, label_tokens: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        query = self.norm1(label_tokens)
        attended, _ = self.attn(query, query, query, attn_mask=attn_bias, need_weights=False)
        label_tokens = label_tokens + attended
        label_tokens = label_tokens + self.mlp(self.norm2(label_tokens))
        return label_tokens


class LabelCorrelationBlock(nn.Module):
    """Self-attention over label tokens for action/reason correlation modeling.

    ``bias_matrix`` is an additive attention prior, normally conditional-log or
    PMI bias built from train-set action/reason co-occurrence. The residual scale
    keeps checkpoint resumes safe: with ``residual_init=0`` the module is an exact
    identity even when new weights are missing from an old checkpoint.
    """

    def __init__(
        self,
        dim: int,
        num_labels: int,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        bias_mode: str = "none",
        bias_matrix: torch.Tensor | None = None,
        bias_weight: float = 0.0,
        residual_init: float = 1.0,
        residual_learnable: bool = True,
    ) -> None:
        super().__init__()
        if bias_mode not in {"none", "cooccur", "pmi"}:
            raise ValueError(f"Unsupported label correlation bias mode: {bias_mode}")
        self.num_labels = int(num_labels)
        self.bias_mode = bias_mode
        self.bias_weight = float(bias_weight)
        if bias_matrix is None:
            bias_matrix = torch.zeros(self.num_labels, self.num_labels)
        bias_matrix = torch.as_tensor(bias_matrix, dtype=torch.float32)
        if tuple(bias_matrix.shape) != (self.num_labels, self.num_labels):
            raise ValueError(
                f"bias_matrix must be [{self.num_labels},{self.num_labels}], got {tuple(bias_matrix.shape)}"
            )
        self.register_buffer("bias_matrix", bias_matrix, persistent=True)
        self.layers = nn.ModuleList(
            [_LabelCorrelationLayer(dim, num_heads, dropout) for _ in range(max(1, int(num_layers)))]
        )
        self.norm = nn.LayerNorm(dim)
        scale = torch.tensor(float(residual_init), dtype=torch.float32)
        if residual_learnable:
            self.residual_scale = nn.Parameter(scale)
        else:
            self.register_buffer("residual_scale", scale, persistent=True)

    def _attention_bias(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.bias_mode == "none" or self.bias_weight == 0.0:
            return None
        return self.bias_matrix.to(device=device, dtype=dtype) * self.bias_weight

    def forward(self, label_tokens: torch.Tensor) -> torch.Tensor:
        if label_tokens.ndim != 3:
            raise ValueError("label_tokens must be [B,L,D]")
        if label_tokens.shape[1] != self.num_labels:
            raise ValueError(f"Expected {self.num_labels} label tokens, got {label_tokens.shape[1]}")
        hidden = label_tokens
        attn_bias = self._attention_bias(label_tokens.device, label_tokens.dtype)
        for layer in self.layers:
            hidden = layer(hidden, attn_bias)
        hidden = self.norm(hidden)
        return label_tokens + self.residual_scale.to(dtype=label_tokens.dtype) * (hidden - label_tokens)
