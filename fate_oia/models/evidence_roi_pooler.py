from __future__ import annotations

from typing import Any

import torch
from torch import nn
from PIL import Image, ImageDraw


def _patch_centers(image_size: tuple[int, int], patch_grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = image_size
    gh, gw = patch_grid
    xs = (torch.arange(gw, dtype=torch.float32) + 0.5) * (float(width) / float(gw))
    ys = (torch.arange(gh, dtype=torch.float32) + 0.5) * (float(height) / float(gh))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.flatten(), yy.flatten()


def box2d_to_patch_weights(box2d: dict[str, Any], *, image_size: tuple[int, int], patch_grid: tuple[int, int]) -> torch.Tensor:
    x1 = float(box2d.get("x1", box2d.get("x_min", 0.0)))
    y1 = float(box2d.get("y1", box2d.get("y_min", 0.0)))
    x2 = float(box2d.get("x2", box2d.get("x_max", x1)))
    y2 = float(box2d.get("y2", box2d.get("y_max", y1)))
    xx, yy = _patch_centers(image_size, patch_grid)
    return ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()


def _vertices(poly2d: dict[str, Any] | list[Any]) -> list[tuple[float, float]]:
    if isinstance(poly2d, dict):
        raw = poly2d.get("vertices") or poly2d.get("points") or []
    else:
        raw = poly2d
    vertices: list[tuple[float, float]] = []
    for point in raw:
        if isinstance(point, dict):
            vertices.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            vertices.append((float(point[0]), float(point[1])))
    return vertices


def poly2d_to_patch_weights(poly2d: dict[str, Any] | list[Any], *, image_size: tuple[int, int], patch_grid: tuple[int, int]) -> torch.Tensor:
    vertices = _vertices(poly2d)
    gh, gw = patch_grid
    if len(vertices) < 3:
        return torch.zeros(gh * gw, dtype=torch.float32)
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(vertices, outline=255, fill=255)
    small = mask.resize((gw, gh), Image.BILINEAR)
    data = torch.tensor(list(small.getdata()), dtype=torch.float32) / 255.0
    return data.clamp(0.0, 1.0)


class EvidenceROIPooler(nn.Module):
    """Pool patch tokens with precomputed box/poly/drivable patch weights."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens: torch.Tensor, roi_weights: torch.Tensor) -> dict[str, torch.Tensor]:
        if patch_tokens.ndim != 3:
            raise ValueError("patch_tokens must be [B,N,D]")
        if roi_weights.ndim != 3:
            raise ValueError("roi_weights must be [B,R,N]")
        if patch_tokens.shape[0] != roi_weights.shape[0] or patch_tokens.shape[1] != roi_weights.shape[2]:
            raise ValueError(f"shape mismatch: tokens={tuple(patch_tokens.shape)} weights={tuple(roi_weights.shape)}")
        denom = roi_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("brn,bnd->brd", roi_weights.float(), patch_tokens.float()) / denom
        valid = roi_weights.sum(dim=-1) > 0
        return {"evidence_tokens": self.norm(pooled.to(dtype=patch_tokens.dtype)), "valid_mask": valid}
