from __future__ import annotations

import torch
import torch.nn.functional as F


def tokens_to_patch_map(tokens: torch.Tensor, image_height: int = 360, image_width: int = 640, patch_size: int = 8) -> torch.Tensor:
    patches = tokens[:, 1:] if tokens.shape[1] == (image_height // patch_size) * (image_width // patch_size) + 1 else tokens
    h = image_height // patch_size
    w = image_width // patch_size
    if patches.shape[1] != h * w:
        side = int(patches.shape[1] ** 0.5)
        h = side
        w = patches.shape[1] // max(side, 1)
    return patches[:, : h * w].transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], h, w)


def sample_reference_points(patch_map: torch.Tensor, refs: torch.Tensor) -> torch.Tensor:
    """Sample [B,R,Q,2] normalized refs from [B,D,H,W] patch map."""
    b, d, _, _ = patch_map.shape
    brq = refs.shape[1] * refs.shape[2]
    grid = refs.clamp(0, 1).mul(2).sub(1).reshape(b, brq, 1, 2)
    sampled = F.grid_sample(patch_map, grid, align_corners=False).squeeze(-1).transpose(1, 2)
    return sampled.reshape(b, refs.shape[1], refs.shape[2], d)
