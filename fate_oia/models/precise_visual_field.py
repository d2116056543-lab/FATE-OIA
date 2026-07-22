from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class VisualFieldBundle:
    action_layers: torch.Tensor
    reason_layers: torch.Tensor
    evidence_layers: torch.Tensor
    action_context: torch.Tensor
    reason_context: torch.Tensor
    evidence_context: torch.Tensor
    cls_tokens: torch.Tensor
    grid_hw: tuple[int, int]


class _FieldAdapter(nn.Module):
    def __init__(self, dim: int, hidden: int, local_kernel: int, rezero_init: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.local = nn.Conv2d(dim, dim, local_kernel, padding=local_kernel // 2, groups=dim)
        self.alpha = nn.Parameter(torch.full((), float(rezero_init)))
        self.beta = nn.Parameter(torch.full((), float(rezero_init)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        batch, count, dim = tokens.shape
        height, width = grid_hw
        if count != height * width:
            raise ValueError("Visual field must retain the entire patch grid")
        low_rank = self.up(F.gelu(self.down(self.norm(tokens))))
        image = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        local = self.local(image).flatten(2).transpose(1, 2)
        return tokens + self.alpha.tanh() * low_rank + self.beta.tanh() * local


class PRECISEVisualField(nn.Module):
    """Three owner-isolated trainable fields applied after frozen DINO tokens."""

    def __init__(self, dim: int = 384, hidden: int = 192, local_kernel: int = 3, rezero_init: float = 0.02, context_pool_hw: tuple[int, int] = (9, 16)) -> None:
        super().__init__()
        self.dim = dim
        self.context_pool_hw = context_pool_hw
        self.action_foundation = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.reason_private = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.evidence_private = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.perspective_projection = nn.Linear(4, dim, bias=False)
        self.layer_embedding = nn.Parameter(torch.randn(3, dim) * 0.01)

    def _perspective(self, device: torch.device, dtype: torch.dtype, grid_hw: tuple[int, int]) -> torch.Tensor:
        height, width = grid_hw
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=dtype), torch.arange(width, device=device, dtype=dtype), indexing="ij")
        lateral = 2.0 * xx / max(width - 1, 1) - 1.0
        distance = -torch.log1p(-yy / max(height - 1, 1) + 1e-5)
        angle = torch.atan((xx - width / 2.0) / (height - yy + 1e-5))
        scale = torch.log1p(yy / max(height - 1, 1))
        coords = torch.stack([lateral, distance, angle, scale], dim=-1).reshape(1, height * width, 4)
        return self.perspective_projection(coords)

    def _context(self, layers: torch.Tensor, cls_tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        batch, layers_count, _, dim = layers.shape
        height, width = grid_hw
        pool_h, pool_w = self.context_pool_hw
        image = layers.reshape(batch * layers_count, height, width, dim).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(image, (pool_h, pool_w)).flatten(2).transpose(1, 2)
        pooled = pooled.reshape(batch, layers_count * pool_h * pool_w, dim)
        return torch.cat([pooled, cls_tokens], dim=1)

    def forward(self, dino_output: dict[str, torch.Tensor | tuple[int, int]]) -> VisualFieldBundle:
        tokens = dino_output["patch_tokens_by_layer"]
        cls_tokens = dino_output["cls_tokens_by_layer"]
        if not isinstance(tokens, torch.Tensor) or not isinstance(cls_tokens, torch.Tensor):
            raise TypeError("Frozen DINO output is malformed")
        grid_hw = dino_output["grid_hw"]
        if not isinstance(grid_hw, tuple):
            raise TypeError("DINO grid is malformed")
        position = self._perspective(tokens.device, tokens.dtype, grid_hw)
        action_layers, reason_layers, evidence_layers = [], [], []
        for layer in range(tokens.shape[1]):
            base = tokens[:, layer] + position + self.layer_embedding[layer].view(1, 1, -1)
            action_layers.append(self.action_foundation[layer](base, grid_hw))
            reason_layers.append(self.reason_private[layer](base.detach(), grid_hw))
            evidence_layers.append(self.evidence_private[layer](base.detach(), grid_hw))
        action = torch.stack(action_layers, dim=1)
        reason = torch.stack(reason_layers, dim=1)
        evidence = torch.stack(evidence_layers, dim=1)
        return VisualFieldBundle(action, reason, evidence, self._context(action, cls_tokens, grid_hw), self._context(reason, cls_tokens, grid_hw), self._context(evidence, cls_tokens, grid_hw), cls_tokens, grid_hw)

    def owned_parameters(self) -> dict[str, list[nn.Parameter]]:
        return {
            "action_foundation": list(self.action_foundation.parameters()) + list(self.perspective_projection.parameters()) + [self.layer_embedding],
            "reason_semantic": list(self.reason_private.parameters()),
            "evidence_core": list(self.evidence_private.parameters()),
        }
