from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor

from .types import InteractVisualOutput


class InteractVisualEncoder(nn.Module):
    """Encode selected observed frames with frozen DINO; never consumes target frames."""

    def __init__(
        self,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        anchor_frames: tuple[int, ...] = (0, 3, 6, 9, 12, 14),
        selected_layers: tuple[int, ...] = (3, 7, 11),
        use_mock_dino: bool = False,
        dim: int = 384,
        dino_chunk_size: int = 2,
        dino_input_height: int = 320,
        dino_input_width: int = 576,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.anchor_frames = tuple(anchor_frames)
        self.dino_input_height = int(dino_input_height)
        self.dino_input_width = int(dino_input_width)
        self.patch_size = int(patch_size)
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
            mock_dim=dim,
        )
        self.dim = self.dino.dim
        self.dino_chunk_size = max(1, int(dino_chunk_size))
        self.fast_motion_cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fast_motion_proj = nn.Linear(64, self.dim)
        self.lowres_motion_proj = nn.Linear(3, self.dim)
        self.temporal_proj = nn.Linear(self.dim * 2, self.dim)

    def forward(self, frames: torch.Tensor) -> InteractVisualOutput:
        if frames.ndim != 5:
            raise ValueError(f"Expected [B,T,3,H,W], got {tuple(frames.shape)}")
        b, t, c, h, w = frames.shape
        if max(self.anchor_frames) >= t:
            raise ValueError(f"anchor frame {max(self.anchor_frames)} outside observed length {t}")
        fast_flat = frames.reshape(b * t, c, h, w)
        fast_motion = self.fast_motion_cnn(fast_flat).flatten(1).reshape(b, t, 64)
        fast_motion_tokens = self.fast_motion_proj(fast_motion)
        lowres_rgb = F.adaptive_avg_pool2d(fast_flat, output_size=(16, 28))
        lowres_motion_maps = self.lowres_motion_proj(lowres_rgb.permute(0, 2, 3, 1)).reshape(b, t, 16, 28, self.dim)
        anchors = frames[:, list(self.anchor_frames)]
        flat = anchors.reshape(b * len(self.anchor_frames), c, h, w)
        dino_size = (self.dino_input_height, self.dino_input_width)
        if flat.shape[-2:] != dino_size:
            flat = F.interpolate(flat, size=dino_size, mode="bilinear", align_corners=False)
        patch_chunks = []
        cls_chunks = []
        for chunk in flat.split(self.dino_chunk_size, dim=0):
            dino = self.dino(chunk)
            patch_chunks.append(dino["patch_tokens_by_layer"])
            cls_chunks.append(dino["cls_tokens_by_layer"])
        patch_all = torch.cat(patch_chunks, dim=0)
        cls_all = torch.cat(cls_chunks, dim=0)
        grid_h = self.dino_input_height // self.patch_size
        grid_w = self.dino_input_width // self.patch_size
        num_patches = grid_h * grid_w
        if patch_all.shape[-2] != num_patches:
            raise RuntimeError(
                f"DINO returned {patch_all.shape[-2]} patches but config implies {num_patches} "
                f"({self.dino_input_height}x{self.dino_input_width}/patch{self.patch_size})"
            )
        patches = patch_all.reshape(b, len(self.anchor_frames), len(self.dino.selected_layers), num_patches, self.dim)
        cls = cls_all.reshape(b, len(self.anchor_frames), len(self.dino.selected_layers), self.dim)
        first_last = torch.cat([patches[:, 0].mean(1), patches[:, -1].mean(1)], dim=-1)
        motion_tokens = self.temporal_proj(first_last)
        anchor_tokens = patches.mean(2).mean(2)
        stats: dict[str, Any] = {
            "anchor_count": len(self.anchor_frames),
            "input_h": h,
            "input_w": w,
            "dino_input_h": self.dino_input_height,
            "dino_input_w": self.dino_input_width,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "dino_grid_h": grid_h,
            "dino_grid_w": grid_w,
            "num_patches": num_patches,
            "formal_target_frame_used": False,
            "dino_chunk_size": self.dino_chunk_size,
            "anchor_indices": list(self.anchor_frames),
        }
        return InteractVisualOutput(
            anchor_tokens=anchor_tokens,
            fast_motion_tokens=fast_motion_tokens,
            lowres_motion_maps=lowres_motion_maps,
            motion_tokens=motion_tokens,
            cls_tokens=cls.mean(2),
            patch_tokens_by_layer=patches,
            grid_hw=(grid_h, grid_w),
            anchor_indices=list(self.anchor_frames),
            stats=stats,
        )
