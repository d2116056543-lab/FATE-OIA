from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VistaScaleSchedule:
    early_scale: float = 0.05
    main_scale: float = 0.15
    late_scale: float = 0.08
    main_start_epoch: int = 3
    late_start_epoch: int = 9

    def max_scale(self, epoch: int) -> float:
        if epoch < self.main_start_epoch:
            return self.early_scale
        if epoch < self.late_start_epoch:
            return self.main_scale
        return self.late_scale


class ACPRLocalGeometricAdapterBlock(nn.Module):
    """Low-rank local-geometric adapter for one DINO patch-token layer."""

    def __init__(self, dim: int = 384, rank: int = 48, grid_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.grid_hw = tuple(grid_hw)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.depthwise = nn.Conv2d(rank, rank, kernel_size=3, padding=1, groups=rank)
        self.up = nn.Linear(rank, dim)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.zeros_(self.depthwise.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, _ = tokens.shape
        h, w = self.grid_hw
        if n != h * w:
            raise ValueError(f"VISTA expects {h*w} patch tokens, got {n}")
        z = F.gelu(self.down(self.norm(tokens)))
        z = z.transpose(1, 2).reshape(b, self.rank, h, w)
        z = F.gelu(self.depthwise(z))
        z = z.flatten(2).transpose(1, 2)
        return self.up(z)


class ACPRPredicateAnchoredVisualAdapter(nn.Module):
    """Predicate-anchored ReZero adapter applied before ACPR downstream heads."""

    def __init__(
        self,
        dim: int = 384,
        rank: int = 48,
        num_layers: int = 3,
        num_predicates: int = 32,
        grid_hw: tuple[int, int] = (45, 80),
        gate_floor: float = 0.20,
        detach_predicate_gate: bool = True,
        schedule: VistaScaleSchedule | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.num_layers = int(num_layers)
        self.num_predicates = int(num_predicates)
        self.grid_hw = tuple(grid_hw)
        self.gate_floor = float(gate_floor)
        self.detach_predicate_gate = bool(detach_predicate_gate)
        self.schedule = schedule or VistaScaleSchedule()
        self.blocks = nn.ModuleList(
            [ACPRLocalGeometricAdapterBlock(dim=dim, rank=rank, grid_hw=grid_hw) for _ in range(num_layers)]
        )
        self.gate_raw = nn.Parameter(torch.zeros(num_layers))
        self.predicate_importance_raw = nn.Parameter(torch.zeros(num_predicates))

    def _predicate_gate(self, probs: torch.Tensor, attention: torch.Tensor) -> torch.Tensor:
        if probs.shape[:2] != attention.shape[:2]:
            raise ValueError("predicate probabilities and attention disagree")
        if self.detach_predicate_gate:
            probs = probs.detach()
            attention = attention.detach()
        importance = F.softplus(self.predicate_importance_raw).view(1, -1, 1)
        raw = (importance * probs.unsqueeze(-1) * attention).sum(dim=1)
        denom = raw.amax(dim=1, keepdim=True).clamp_min(1e-6)
        norm = (raw / denom).clamp(0.0, 1.0)
        return self.gate_floor + (1.0 - self.gate_floor) * norm

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        raw_predicate_probs: torch.Tensor,
        raw_predicate_attention: torch.Tensor,
        epoch: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float | list[float]]]:
        b, layers, n, d = patch_tokens_by_layer.shape
        if layers != self.num_layers:
            raise ValueError(f"VISTA configured for {self.num_layers} layers, got {layers}")
        if d != self.dim:
            raise ValueError(f"VISTA configured for dim {self.dim}, got {d}")
        gate = self._predicate_gate(raw_predicate_probs, raw_predicate_attention).to(patch_tokens_by_layer.dtype)
        max_scale = self.schedule.max_scale(int(epoch))
        alpha = max_scale * torch.tanh(self.gate_raw).to(patch_tokens_by_layer.dtype)
        adapted_layers: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        delta_maps: list[torch.Tensor] = []
        for idx, block in enumerate(self.blocks):
            z = block(patch_tokens_by_layer[:, idx])
            delta = alpha[idx].view(1, 1, 1) * gate.unsqueeze(-1) * z
            adapted_layers.append(patch_tokens_by_layer[:, idx] + delta)
            delta_norm = delta.norm(dim=-1)
            delta_norms.append(delta_norm.mean())
            delta_maps.append(delta_norm.detach())
        adapted = torch.stack(adapted_layers, dim=1)
        delta_stack = torch.stack(delta_maps, dim=1)
        high = gate >= gate.quantile(0.75, dim=1, keepdim=True)
        high_mass = (delta_stack.mean(1) * high.float()).sum(1) / delta_stack.mean(1).sum(1).clamp_min(1e-6)
        entropy = -(gate.clamp_min(1e-9).log() * gate).mean()
        stats = {
            "vista_enabled": True,
            "vista_alpha_per_layer": alpha.detach(),
            "vista_alpha_abs_mean": alpha.abs().mean().detach(),
            "vista_adapter_delta_norm_per_layer": torch.stack(delta_norms).detach(),
            "vista_adapter_delta_norm_mean": torch.stack(delta_norms).mean().detach(),
            "vista_gate_map": gate.detach(),
            "vista_gate_mean": gate.mean().detach(),
            "vista_gate_max": gate.max().detach(),
            "vista_gate_entropy": entropy.detach(),
            "vista_delta_mass_on_high_gate": high_mass.mean().detach(),
            "vista_delta_uniformity": (delta_stack.mean(1).std(dim=1) / delta_stack.mean(1).mean(dim=1).clamp_min(1e-6)).mean().detach(),
        }
        return adapted, stats

