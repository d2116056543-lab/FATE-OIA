from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AIECertDeformableReread(nn.Module):
    def __init__(self, dim=384, grid_hw=(45, 80), num_layers=3, points_per_layer=8, max_offset=0.25):
        super().__init__()
        self.grid_hw, self.num_layers, self.points_per_layer = grid_hw, num_layers, points_per_layer
        self.max_offset = float(max_offset)
        self.global_projection = nn.Linear(dim, dim)
        self.map_projection = nn.Linear(dim, dim)
        self.query_norm = nn.LayerNorm(dim)
        self.offset_head = nn.Linear(dim, num_layers * points_per_layer * 2)
        self.weight_head = nn.Linear(dim, num_layers * points_per_layer)

    def _coords(self, device, dtype):
        h, w = self.grid_hw
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=device, dtype=dtype),
                                torch.linspace(0, 1, w, device=device, dtype=dtype), indexing="ij")
        return torch.stack((xx, yy), -1).reshape(h * w, 2)

    def forward(self, probes: Tensor, field: Tensor, map_pre: Tensor, global_token: Tensor) -> dict[str, Tensor]:
        b, actions, atoms, dim = probes.shape
        layers, patches = field.shape[1:3]
        coords = self._coords(field.device, field.dtype)
        reference = torch.einsum("bakn,nc->bakc", map_pre, coords)
        mixed_field = field.mean(1)
        map_summary = torch.einsum("bakn,bnd->bakd", map_pre, mixed_field)
        local_query = self.query_norm(probes + self.global_projection(global_token) + self.map_projection(map_summary))
        offsets = torch.tanh(self.offset_head(local_query)).view(b, actions, atoms, layers, self.points_per_layer, 2)
        offsets = offsets * self.max_offset
        weights = torch.softmax(self.weight_head(local_query).view(b, actions, atoms, -1), -1)
        weights = weights.view(b, actions, atoms, layers, self.points_per_layer)
        grid = (reference[..., None, None, :] + offsets).clamp(0, 1).mul(2).sub(1)
        sampled = []
        h, w = self.grid_hw
        for layer in range(layers):
            source = field[:, layer].transpose(1, 2).reshape(b, dim, h, w)
            layer_grid = grid[:, :, :, layer].reshape(b, actions * atoms * self.points_per_layer, 1, 2)
            value = F.grid_sample(source, layer_grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled.append(value.squeeze(-1).transpose(1, 2).reshape(b, actions, atoms, self.points_per_layer, dim))
        local = (torch.stack(sampled, 3) * weights[..., None]).sum((3, 4))
        return {"local_query": local_query, "local_token": local, "reference_point": reference,
                "sampling_offsets": offsets, "sampling_weights": weights}
