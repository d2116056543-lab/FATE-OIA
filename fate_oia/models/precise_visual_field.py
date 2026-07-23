from __future__ import annotations

from dataclasses import dataclass
import math

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
        normalized = self.norm(tokens)
        low_rank = self.up(F.gelu(self.down(normalized)))
        image = tokens.transpose(1, 2).reshape(batch, dim, height, width)
        local = self.local(image).flatten(2).transpose(1, 2)
        return normalized + self.alpha.tanh() * low_rank + self.beta.tanh() * local


class PRECISEVisualField(nn.Module):
    """Three owner-isolated trainable fields applied after frozen DINO tokens."""

    def __init__(self, dim: int = 384, hidden: int = 192, local_kernel: int = 3, rezero_init: float = 0.02, context_pool_hw: tuple[int, int] = (9, 16)) -> None:
        super().__init__()
        self.dim = dim
        self.context_pool_hw = context_pool_hw
        self.action_foundation = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.reason_private = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.evidence_private = nn.ModuleList([_FieldAdapter(dim, hidden, local_kernel, rezero_init) for _ in range(3)])
        self.perspective_projection = nn.ModuleDict({owner: nn.Linear(4, dim, bias=False) for owner in ("action", "reason", "evidence")})
        self.layer_embedding = nn.ParameterDict({owner: nn.Parameter(torch.randn(3, dim) * 0.01) for owner in ("action", "reason", "evidence")})

    def _perspective(self, owner: str, device: torch.device, dtype: torch.dtype, grid_hw: tuple[int, int]) -> torch.Tensor:
        height, width = grid_hw
        # Build perspective coordinates in fp32. In bf16, the 1e-5 bottom-row
        # guard rounds away and log1p(-1) produces an infinite feature.
        yy, xx = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32), torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        lateral = 2.0 * xx / max(width - 1, 1) - 1.0
        normalized_y = (yy / max(height - 1, 1)).clamp(0.0, 1.0 - 1e-5)
        distance = -torch.log1p(-normalized_y)
        angle = torch.atan((xx - width / 2.0) / (height - yy + 1e-5))
        scale = torch.log1p(yy / max(height - 1, 1))
        coords = torch.stack([lateral, distance, angle, scale], dim=-1).reshape(1, height * width, 4)
        return (self._sinusoidal_2d(device, dtype, grid_hw) + self.perspective_projection[owner](coords).to(dtype=dtype))

    def _sinusoidal_2d(self, device: torch.device, dtype: torch.dtype, grid_hw: tuple[int, int]) -> torch.Tensor:
        height, width = grid_hw
        if self.dim % 4:
            raise ValueError("PRECISE 2D positional encoding requires dim divisible by four")
        quarter = self.dim // 4
        frequency = torch.exp(torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(quarter - 1, 1)))
        y, x = torch.meshgrid(torch.arange(height, device=device, dtype=torch.float32), torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
        x_phase = x.reshape(-1, 1) * frequency.reshape(1, -1)
        y_phase = y.reshape(-1, 1) * frequency.reshape(1, -1)
        return torch.cat([x_phase.sin(), x_phase.cos(), y_phase.sin(), y_phase.cos()], dim=-1).unsqueeze(0).to(dtype=dtype)

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
        positions = {owner: self._perspective(owner, tokens.device, tokens.dtype, grid_hw) for owner in ("action", "reason", "evidence")}
        action_layers, reason_layers, evidence_layers = [], [], []
        for layer in range(tokens.shape[1]):
            raw = tokens[:, layer]
            action_layers.append(self.action_foundation[layer](raw, grid_hw) + positions["action"] + self.layer_embedding["action"][layer].view(1, 1, -1))
            reason_layers.append(self.reason_private[layer](raw.detach(), grid_hw) + positions["reason"] + self.layer_embedding["reason"][layer].view(1, 1, -1))
            evidence_layers.append(self.evidence_private[layer](raw.detach(), grid_hw) + positions["evidence"] + self.layer_embedding["evidence"][layer].view(1, 1, -1))
        action = torch.stack(action_layers, dim=1)
        reason = torch.stack(reason_layers, dim=1)
        evidence = torch.stack(evidence_layers, dim=1)
        return VisualFieldBundle(action, reason, evidence, self._context(action, cls_tokens, grid_hw), self._context(reason, cls_tokens, grid_hw), self._context(evidence, cls_tokens, grid_hw), cls_tokens, grid_hw)

    def owned_parameters(self) -> dict[str, list[nn.Parameter]]:
        return {
            "action_foundation": list(self.action_foundation.parameters()) + list(self.perspective_projection["action"].parameters()) + [self.layer_embedding["action"]],
            "reason_semantic": list(self.reason_private.parameters()) + list(self.perspective_projection["reason"].parameters()) + [self.layer_embedding["reason"]],
            "evidence_core": list(self.evidence_private.parameters()) + list(self.perspective_projection["evidence"].parameters()) + [self.layer_embedding["evidence"]],
        }
