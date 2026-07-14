from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


class MOSAICWeakMultiView:
    """Create reproducible weak views with invertible flip metadata.

    Coordinates are expected in the normalized ``[-1, 1]`` convention used by
    the geometry sampler, so a horizontal flip negates their x component.
    """

    _ACTION_FLIP_PERMUTATION = (0, 1, 3, 2)
    _REASON_FLIP_PERMUTATION = (0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19, 20, 9, 10, 11, 12, 13, 14)

    def __init__(
        self,
        factor_names: Sequence[str],
        *,
        mirror_pairs: Mapping[str, str] | None = None,
        flip_probability: float = 0.5,
        brightness_jitter: float = 0.10,
        contrast_jitter: float = 0.10,
        seed: int = 0,
    ) -> None:
        self.factor_names = tuple(factor_names)
        if not self.factor_names or len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError("multiview requires unique non-empty factor names")
        if any(not isinstance(name, str) or not name for name in self.factor_names):
            raise ValueError("multiview factor names must be non-empty strings")
        if not 0.0 <= flip_probability <= 1.0:
            raise ValueError("flip_probability must be in [0,1]")
        if not 0.0 <= brightness_jitter <= 0.20 or not 0.0 <= contrast_jitter <= 0.20:
            raise ValueError("weak brightness and contrast jitter must be in [0,0.20]")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")

        self.flip_probability = float(flip_probability)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)
        self._generator = torch.Generator().manual_seed(seed)
        self._flip_permutation = self._build_flip_permutation(mirror_pairs)

    def _build_flip_permutation(self, mirror_pairs: Mapping[str, str] | None) -> tuple[int, ...]:
        name_to_index = {name: index for index, name in enumerate(self.factor_names)}
        pairs: dict[str, str] = {}

        def add_pair(name: str, partner: str) -> None:
            if name not in name_to_index or partner not in name_to_index:
                raise ValueError("mirror pairs must reference known factor names")
            if pairs.get(name, partner) != partner or pairs.get(partner, name) != name:
                raise ValueError("mirror pairs must be symmetric and non-conflicting")
            pairs[name] = partner
            pairs[partner] = name

        for name, partner in (mirror_pairs or {}).items():
            add_pair(name, partner)
        for name in self.factor_names:
            if "left" in name:
                partner = name.replace("left", "right")
            elif "right" in name:
                partner = name.replace("right", "left")
            else:
                continue
            if partner in name_to_index:
                add_pair(name, partner)

        permutation = list(range(len(self.factor_names)))
        for name, partner in pairs.items():
            permutation[name_to_index[name]] = name_to_index[partner]
        return tuple(permutation)

    def _draw_metadata(self, *, allow_horizontal_flip: bool) -> dict[str, Any]:
        horizontal_flip = allow_horizontal_flip and bool(
            torch.rand((), generator=self._generator).item() < self.flip_probability
        )
        brightness_delta = (2.0 * torch.rand((), generator=self._generator).item() - 1.0) * self.brightness_jitter
        contrast_factor = 1.0 + (2.0 * torch.rand((), generator=self._generator).item() - 1.0) * self.contrast_jitter
        return {
            "horizontal_flip": horizontal_flip,
            "brightness_delta": brightness_delta,
            "contrast_factor": contrast_factor,
            "hue_delta": 0.0,
            "factor_permutation": self._flip_permutation if horizontal_flip else tuple(range(len(self.factor_names))),
            "action_permutation": self._ACTION_FLIP_PERMUTATION if horizontal_flip else tuple(range(4)),
            "reason_permutation": self._REASON_FLIP_PERMUTATION if horizontal_flip else tuple(range(21)),
            "coordinate_system": "normalized_minus_one_to_one",
        }

    @staticmethod
    def _apply_weak_photometric_transform(image: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        transformed = image.flip(-1) if metadata["horizontal_flip"] else image
        mean = transformed.mean(dim=(-2, -1), keepdim=True)
        transformed = (transformed - mean) * float(metadata["contrast_factor"]) + mean
        transformed = transformed + float(metadata["brightness_delta"])
        # The formal direct-image loader already applies DINO normalization.
        # Clamping such tensors to [0,1] destroys the input distribution.
        if image.detach().amin() >= 0.0 and image.detach().amax() <= 1.0:
            transformed = transformed.clamp(0.0, 1.0)
        return transformed

    def __call__(self, image: torch.Tensor) -> dict[str, Any]:
        if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("multiview expects a floating [3,H,W] image tensor")
        if not image.is_floating_point() or image.shape[-2] <= 0 or image.shape[-1] <= 0:
            raise ValueError("multiview expects a floating [3,H,W] image tensor")

        # The first view is the canonical label space; only the second view may flip.
        metadata = (
            self._draw_metadata(allow_horizontal_flip=False),
            self._draw_metadata(allow_horizontal_flip=True),
        )
        images = tuple(self._apply_weak_photometric_transform(image, item) for item in metadata)
        return {
            "images": images,
            "metadata": metadata,
            "invert_factor_masks": self.invert_factor_masks,
            "invert_factor_coordinates": self.invert_factor_coordinates,
        }

    @staticmethod
    def _canonical_dim(ndim: int, dim: int) -> int:
        if not -ndim <= dim < ndim:
            raise ValueError("factor_dim is outside the tensor rank")
        return dim % ndim

    def _restore_factor_axis(self, values: torch.Tensor, metadata: Mapping[str, Any], factor_dim: int) -> torch.Tensor:
        factor_dim = self._canonical_dim(values.ndim, factor_dim)
        permutation = torch.as_tensor(metadata["factor_permutation"], device=values.device, dtype=torch.long)
        if values.shape[factor_dim] != len(self.factor_names) or permutation.numel() != len(self.factor_names):
            raise ValueError("factor axis does not match multiview factor metadata")
        return torch.empty_like(values).index_copy(factor_dim, permutation, values)

    def _restore_label_axis(
        self,
        values: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        key: str,
        label_dim: int,
        axis: int,
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError("label values must expose a label axis")
        axis = self._canonical_dim(values.ndim, axis)
        permutation = torch.as_tensor(metadata[key], device=values.device, dtype=torch.long)
        if values.shape[axis] != label_dim or permutation.numel() != label_dim:
            raise ValueError("label axis does not match multiview metadata")
        return torch.empty_like(values).index_copy(axis, permutation, values)

    def invert_factor_masks(
        self,
        masks: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        factor_dim: int = -3,
    ) -> torch.Tensor:
        if not isinstance(masks, torch.Tensor) or masks.ndim < 3:
            raise ValueError("factor masks must have factor, height, and width axes")
        restored = masks.flip(-1) if metadata["horizontal_flip"] else masks
        return self._restore_factor_axis(restored, metadata, factor_dim)

    def invert_factor_values(
        self,
        values: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        factor_dim: int = -1,
    ) -> torch.Tensor:
        """Restore the ontology order for non-spatial per-factor values."""
        if not isinstance(values, torch.Tensor) or values.ndim < 1:
            raise ValueError("factor values must expose a factor axis")
        return self._restore_factor_axis(values, metadata, factor_dim)

    def invert_action_values(
        self,
        values: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        action_dim: int = -1,
    ) -> torch.Tensor:
        """Restore the four action scores to the canonical driving label order."""
        return self._restore_label_axis(
            values,
            metadata,
            key="action_permutation",
            label_dim=4,
            axis=action_dim,
        )

    def invert_reason_values(
        self,
        values: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        reason_dim: int = -1,
    ) -> torch.Tensor:
        """Restore directional reason scores, including the 9<->15 through 14<->20 pairs."""
        return self._restore_label_axis(
            values,
            metadata,
            key="reason_permutation",
            label_dim=21,
            axis=reason_dim,
        )

    def invert_factor_coordinates(
        self,
        coordinates: torch.Tensor,
        metadata: Mapping[str, Any],
        *,
        factor_dim: int = -2,
    ) -> torch.Tensor:
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim < 2 or coordinates.shape[-1] != 2:
            raise ValueError("factor coordinates must end in an xy axis")
        restored = coordinates.clone()
        if metadata["horizontal_flip"]:
            restored[..., 0].neg_()
        return self._restore_factor_axis(restored, metadata, factor_dim)

    @staticmethod
    def invert_geometry_coordinates(coordinates: torch.Tensor, metadata: Mapping[str, Any]) -> torch.Tensor:
        """Restore geometry coordinates whose final axis is canonical ``(x, y)``."""
        if not isinstance(coordinates, torch.Tensor) or coordinates.ndim < 1 or coordinates.shape[-1] != 2:
            raise ValueError("geometry coordinates must end in an xy axis")
        restored = coordinates.clone()
        if metadata["horizontal_flip"]:
            restored[..., 0].neg_()
        return restored
