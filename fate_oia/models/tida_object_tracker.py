from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TIDAFrozenPointTracker(nn.Module):
    """Frozen CoTracker adapter with safe batching and normalized coordinates."""

    def __init__(self, predictor: nn.Module, *, grid_size: int = 8) -> None:
        super().__init__()
        if grid_size < 2:
            raise ValueError("grid_size must be at least two")
        self.predictor = predictor.eval()
        self.grid_size = int(grid_size)
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        )
        for parameter in self.predictor.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_local_repository(
        cls,
        repository: str | Path,
        *,
        model_name: str = "cotracker3_offline",
        grid_size: int = 8,
    ) -> "TIDAFrozenPointTracker":
        repository = Path(repository)
        if not repository.exists():
            raise FileNotFoundError(f"CoTracker repository does not exist: {repository}")
        predictor = torch.hub.load(str(repository), model_name, source="local")
        return cls(predictor, grid_size=grid_size)

    def train(self, mode: bool = True):
        super().train(False)
        self.predictor.eval()
        return self

    def forward(self, normalized_video: torch.Tensor) -> dict[str, torch.Tensor]:
        if normalized_video.ndim != 5 or normalized_video.shape[2] != 3:
            raise ValueError("normalized_video must be [B,T,3,H,W]")
        video = (
            normalized_video * self.image_std.to(normalized_video.dtype)
            + self.image_mean.to(normalized_video.dtype)
        ).clamp(0.0, 1.0) * 255.0
        tracks_by_sample = []
        visibility_by_sample = []
        # CoTracker3 offline uses view() on an expanded coordinate tensor for
        # B>1. Sequential predictor calls preserve official semantics without
        # patching the dependency and still batch the downstream transport.
        with torch.no_grad():
            for sample in video.split(1, dim=0):
                tracks, visibility = self.predictor(
                    sample,
                    grid_size=self.grid_size,
                    grid_query_frame=0,
                    backward_tracking=True,
                )
                tracks_by_sample.append(tracks)
                visibility_by_sample.append(visibility)
        tracks = torch.cat(tracks_by_sample, dim=0)
        visibility = torch.cat(visibility_by_sample, dim=0) > 0.5
        height, width = normalized_video.shape[-2:]
        xy = tracks.to(normalized_video.dtype).clone()
        xy[..., 0] = 2.0 * xy[..., 0] / max(width - 1, 1) - 1.0
        xy[..., 1] = 2.0 * xy[..., 1] / max(height - 1, 1) - 1.0
        return {
            "object_tracks_xy": xy,
            "object_tracks_visibility": visibility,
            "object_tracks_visibility_rate": visibility.float().mean((1, 2)),
        }
