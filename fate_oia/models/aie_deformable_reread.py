from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AIEDeformableReread(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        grid_hw: tuple[int, int] = (45, 80),
        num_layers: int = 3,
        points_per_layer: int = 8,
        max_offset: float = 0.25,
    ) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.num_layers = num_layers
        self.points_per_layer = points_per_layer
        self.max_offset = float(max_offset)
        self.offset_head = nn.Linear(dim, num_layers * points_per_layer * 2)
        self.weight_head = nn.Linear(dim, num_layers * points_per_layer)

    def _coordinates(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        h, w = self.grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).reshape(h * w, 2)

    def forward(self, probes: Tensor, field: Tensor, evidence_map: Tensor) -> dict[str, Tensor]:
        b, action_dim, probe_count, dim = probes.shape
        _, layers, patches, field_dim = field.shape
        if layers != self.num_layers or field_dim != dim:
            raise ValueError("AIE deformable field shape does not match module contract")
        if patches != self.grid_hw[0] * self.grid_hw[1]:
            raise ValueError("AIE deformable reread requires the configured 2D patch grid")
        coords = self._coordinates(field.device, field.dtype)
        reference = torch.einsum("bakn,nc->bakc", evidence_map, coords)
        offsets = torch.tanh(self.offset_head(probes)).view(
            b, action_dim, probe_count, layers, self.points_per_layer, 2
        ) * self.max_offset
        weights = torch.softmax(
            self.weight_head(probes).view(b, action_dim, probe_count, layers * self.points_per_layer),
            dim=-1,
        ).view(b, action_dim, probe_count, layers, self.points_per_layer)
        sample_xy = (reference[..., None, None, :] + offsets).clamp(0, 1)
        grid = sample_xy.mul(2).sub(1)
        sampled_layers = []
        h, w = self.grid_hw
        for layer in range(layers):
            source = field[:, layer].transpose(1, 2).reshape(b, dim, h, w)
            layer_grid = grid[:, :, :, layer].reshape(b, action_dim * probe_count * self.points_per_layer, 1, 2)
            sampled = F.grid_sample(source, layer_grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, action_dim, probe_count, self.points_per_layer, dim)
            sampled_layers.append(sampled)
        sampled_all = torch.stack(sampled_layers, dim=3)
        local = (sampled_all * weights[..., None]).sum(dim=(3, 4))
        return {
            "local_token": local,
            "reference_point": reference,
            "sampling_offsets": offsets,
            "sampling_weights": weights,
        }


