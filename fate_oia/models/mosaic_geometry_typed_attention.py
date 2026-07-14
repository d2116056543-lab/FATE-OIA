from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class MOSAICGeometryTypedAttention(nn.Module):
    """Vectorized point/object, curve, and region sampling around soft anchors."""

    _SUPPORTED_TYPES = {"point", "object", "curve", "region"}

    def __init__(
        self,
        factor_types: Sequence[str],
        *,
        dim: int = 384,
        anchors_per_factor: int = 2,
        heads: int = 4,
        point_samples: int = 4,
        curve_samples: int = 16,
        region_samples: int = 12,
    ) -> None:
        super().__init__()
        factor_types = tuple(factor_types)
        if not factor_types:
            raise ValueError("typed attention requires at least one factor")
        if any(factor_type not in self._SUPPORTED_TYPES for factor_type in factor_types):
            raise ValueError("typed attention received an unsupported factor type")
        for value, name in (
            (dim, "dim"),
            (anchors_per_factor, "anchors_per_factor"),
            (heads, "heads"),
            (point_samples, "point_samples"),
            (curve_samples, "curve_samples"),
            (region_samples, "region_samples"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if anchors_per_factor != 2 or heads != 4 or region_samples != 12:
            raise ValueError("MOSAIC typed attention requires M=2, H=4, and a 3x4 region grid")
        if (point_samples, curve_samples) not in {(4, 16), (8, 12)}:
            raise ValueError("MOSAIC typed attention supports IC-DOR 4/16 or legacy 8/12 sampling")

        self.factor_types = factor_types
        self.dim = dim
        self.anchors_per_factor = anchors_per_factor
        self.heads = heads
        self.point_samples = point_samples
        self.curve_samples = curve_samples
        self.region_samples = region_samples
        self.max_samples = max(point_samples, curve_samples, region_samples)

        point_indices = [index for index, factor_type in enumerate(factor_types) if factor_type in {"point", "object"}]
        curve_indices = [index for index, factor_type in enumerate(factor_types) if factor_type == "curve"]
        region_indices = [index for index, factor_type in enumerate(factor_types) if factor_type == "region"]
        self.register_buffer("point_indices", torch.tensor(point_indices, dtype=torch.long), persistent=False)
        self.register_buffer("curve_indices", torch.tensor(curve_indices, dtype=torch.long), persistent=False)
        self.register_buffer("region_indices", torch.tensor(region_indices, dtype=torch.long), persistent=False)

        point_base = self._build_point_offsets(heads, point_samples)
        self.register_buffer("point_offset_base", point_base, persistent=True)
        self.point_offset_delta = nn.Parameter(torch.zeros(len(point_indices), heads, point_samples, 2))

        tangent = torch.zeros(len(curve_indices), heads, 2)
        tangent[..., 1] = 1.0
        self.curve_tangent_raw = nn.Parameter(tangent)
        self.register_buffer(
            "curve_longitudinal_base",
            torch.linspace(-0.18, 0.18, curve_samples).view(1, 1, curve_samples),
            persistent=True,
        )
        self.register_buffer(
            "curve_lateral_base",
            torch.linspace(-0.045, 0.045, heads).view(1, heads, 1).expand(1, heads, curve_samples).clone(),
            persistent=True,
        )
        self.curve_longitudinal_delta = nn.Parameter(torch.zeros(len(curve_indices), heads, curve_samples))
        self.curve_lateral_delta = nn.Parameter(torch.zeros(len(curve_indices), heads, curve_samples))
        self.register_buffer(
            "curve_arc_length_position",
            torch.linspace(-1.0, 1.0, curve_samples).view(1, curve_samples, 1),
            persistent=True,
        )
        curve_encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=2 * dim,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.curve_sequence_encoder = nn.TransformerEncoder(curve_encoder_layer, num_layers=2)
        # Kept explicitly for audit/tests across PyTorch versions.
        self.curve_sequence_encoder.num_layers = 2

        target_extent = 0.18
        extent_fraction = (target_extent - 0.05) / (0.35 - 0.05)
        extent_logit = math.log(extent_fraction / (1.0 - extent_fraction))
        self.region_extent_raw = nn.Parameter(torch.full((len(region_indices), heads, 2), extent_logit))
        self.register_buffer("region_grid_base", self._build_region_grid(), persistent=True)
        self.region_grid_delta = nn.Parameter(torch.zeros(len(region_indices), heads, region_samples, 2))

        valid_counts = [point_samples if value in {"point", "object"} else curve_samples if value == "curve" else region_samples for value in factor_types]
        valid_mask = torch.arange(self.max_samples).unsqueeze(0) < torch.tensor(valid_counts).unsqueeze(1)
        self.register_buffer("sample_valid_mask", valid_mask, persistent=True)

    @staticmethod
    def _build_point_offsets(heads: int, samples: int) -> torch.Tensor:
        sample_angles = torch.arange(samples, dtype=torch.float32) * (2.0 * math.pi / samples)
        head_phase = torch.arange(heads, dtype=torch.float32).unsqueeze(1) * (math.pi / max(heads * samples, 1))
        angles = sample_angles.unsqueeze(0) + head_phase
        radius = torch.linspace(0.025, 0.12, samples).unsqueeze(0)
        return torch.stack((torch.cos(angles) * radius, torch.sin(angles) * radius), dim=-1)

    @staticmethod
    def _build_region_grid() -> torch.Tensor:
        vertical, horizontal = torch.meshgrid(
            torch.linspace(-1.0, 1.0, 3),
            torch.linspace(-1.0, 1.0, 4),
            indexing="ij",
        )
        return torch.stack((horizontal.reshape(-1), vertical.reshape(-1)), dim=-1)

    @staticmethod
    def _sample_geometry_group(
        sampling_feature_map: torch.Tensor,
        coordinates: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, factor_count, anchors, heads, samples, _ = coordinates.shape
        flat_grid = coordinates.reshape(batch_size, factor_count * anchors * heads, samples, 2)
        with torch.autocast(device_type=sampling_feature_map.device.type, enabled=False):
            sampled = F.grid_sample(
                sampling_feature_map,
                flat_grid.to(dtype=sampling_feature_map.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        sampled = sampled.to(dtype=output_dtype)
        return sampled.permute(0, 2, 3, 1).reshape(
            batch_size,
            factor_count,
            anchors,
            heads,
            samples,
            sampling_feature_map.shape[1],
        )

    def _point_coordinates(self, anchors: torch.Tensor) -> torch.Tensor:
        offsets = self.point_offset_base.float().unsqueeze(0) + 0.08 * torch.tanh(self.point_offset_delta.float())
        return (anchors[:, :, :, None, None, :] + offsets[None, :, None]).clamp(-1.0, 1.0)

    def _curve_coordinates(self, anchors: torch.Tensor) -> torch.Tensor:
        tangent = F.normalize(self.curve_tangent_raw.float(), dim=-1, eps=1e-6)
        normal = torch.stack((-tangent[..., 1], tangent[..., 0]), dim=-1)
        longitudinal = self.curve_longitudinal_base.float() + 0.08 * torch.tanh(self.curve_longitudinal_delta.float())
        lateral = self.curve_lateral_base.float() + 0.05 * torch.tanh(self.curve_lateral_delta.float())
        offsets = tangent[..., None, :] * longitudinal[..., None] + normal[..., None, :] * lateral[..., None]
        return (anchors[:, :, :, None, None, :] + offsets[None, :, None]).clamp(-1.0, 1.0)

    def _region_coordinates(self, anchors: torch.Tensor) -> torch.Tensor:
        extent = 0.05 + 0.30 * torch.sigmoid(self.region_extent_raw.float())
        grid = self.region_grid_base.float()[None, None] + 0.10 * torch.tanh(self.region_grid_delta.float())
        offsets = grid * extent[..., None, :]
        return (anchors[:, :, :, None, None, :] + offsets[None, :, None]).clamp(-1.0, 1.0)

    def _pad_samples(self, values: torch.Tensor) -> torch.Tensor:
        padding = self.max_samples - values.shape[-2]
        return F.pad(values, (0, 0, 0, padding)) if padding else values

    def _encode_curve_sequences(self, sampled_features: torch.Tensor) -> torch.Tensor:
        """Run the ordered, per-factor curve samples through a local 1D encoder."""
        batch_size, factor_count, anchors, heads, samples, dim = sampled_features.shape
        if samples != self.curve_samples:
            raise ValueError("curve sequence encoder received an invalid sample count")
        sequence = sampled_features.reshape(batch_size * factor_count * anchors * heads, samples, dim)
        sequence = sequence + self.curve_arc_length_position.to(dtype=sequence.dtype, device=sequence.device)
        encoded = self.curve_sequence_encoder(sequence)
        return encoded.reshape(batch_size, factor_count, anchors, heads, samples, dim)

    def forward(self, feature_map: torch.Tensor, anchor_coordinates: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = feature_map.shape[0] if feature_map.ndim > 0 else -1
        expected_feature_shape = (batch_size, self.dim, 45, 80)
        expected_anchor_shape = (batch_size, len(self.factor_types), self.anchors_per_factor, 2)
        if tuple(feature_map.shape) != expected_feature_shape or tuple(anchor_coordinates.shape) != expected_anchor_shape:
            raise ValueError("MOSAIC typed attention shape contract requires [B,D,45,80] and [B,F,M,2]")
        if not feature_map.is_floating_point() or not anchor_coordinates.is_floating_point():
            raise ValueError("MOSAIC typed attention requires floating-point tensors")
        if feature_map.device != anchor_coordinates.device:
            raise ValueError("feature map and anchors must share a device")

        output_dtype = feature_map.dtype
        compute_dtype = torch.float64 if output_dtype == torch.float64 else torch.float32
        sampling_feature_map = feature_map.to(dtype=compute_dtype)

        coordinates = torch.zeros(
            batch_size,
            len(self.factor_types),
            self.anchors_per_factor,
            self.heads,
            self.max_samples,
            2,
            device=feature_map.device,
            dtype=torch.float32,
        )
        sampled_features = feature_map.new_zeros(
            batch_size,
            len(self.factor_types),
            self.anchors_per_factor,
            self.heads,
            self.max_samples,
            self.dim,
        )

        if self.point_indices.numel():
            point_anchors = anchor_coordinates.index_select(1, self.point_indices).float()
            point_coordinates = self._point_coordinates(point_anchors)
            point_samples = self._sample_geometry_group(sampling_feature_map, point_coordinates, output_dtype)
            coordinates = coordinates.index_copy(1, self.point_indices, self._pad_samples(point_coordinates))
            sampled_features = sampled_features.index_copy(1, self.point_indices, self._pad_samples(point_samples))
        if self.curve_indices.numel():
            curve_anchors = anchor_coordinates.index_select(1, self.curve_indices).float()
            curve_coordinates = self._curve_coordinates(curve_anchors)
            curve_samples = self._sample_geometry_group(sampling_feature_map, curve_coordinates, output_dtype)
            curve_samples = self._encode_curve_sequences(curve_samples)
            coordinates = coordinates.index_copy(1, self.curve_indices, self._pad_samples(curve_coordinates))
            sampled_features = sampled_features.index_copy(1, self.curve_indices, self._pad_samples(curve_samples))
        if self.region_indices.numel():
            region_anchors = anchor_coordinates.index_select(1, self.region_indices).float()
            region_coordinates = self._region_coordinates(region_anchors)
            region_samples = self._sample_geometry_group(sampling_feature_map, region_coordinates, output_dtype)
            coordinates = coordinates.index_copy(1, self.region_indices, self._pad_samples(region_coordinates))
            sampled_features = sampled_features.index_copy(1, self.region_indices, self._pad_samples(region_samples))

        return {
            "sampled_features": sampled_features,
            "sampling_coordinates": coordinates,
            "sample_valid_mask": self.sample_valid_mask,
        }
