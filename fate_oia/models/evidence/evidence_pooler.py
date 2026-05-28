from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.evidence_roi_pooler import box2d_to_patch_weights, poly2d_to_patch_weights


def _image_size(meta: dict[str, Any]) -> tuple[int, int]:
    width = int(meta.get("resized_width") or meta.get("image_width") or meta.get("width") or 640)
    height = int(meta.get("resized_height") or meta.get("image_height") or meta.get("height") or 360)
    return width, height


def map_image_xy_to_patch_index(x: float, y: float, transform_meta: dict[str, Any], patch_size: int = 8) -> int:
    width, height = _image_size(transform_meta)
    gw = int(transform_meta.get("patch_grid_w") or max(1, round(width / patch_size)))
    gh = int(transform_meta.get("patch_grid_h") or max(1, round(height / patch_size)))
    px = min(gw - 1, max(0, int(float(x) / max(1.0, width) * gw)))
    py = min(gh - 1, max(0, int(float(y) / max(1.0, height) * gh)))
    return py * gw + px


def masked_average_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float()
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return torch.einsum("brn,bnd->brd", weights, tokens.float()) / denom


class EvidenceTokenPooler(nn.Module):
    def __init__(self, dim: int = 384, max_object_tokens: int = 24, max_lane_tokens: int = 8, max_drivable_tokens: int = 4, num_categories: int = 64) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_tokens = int(max_object_tokens + max_lane_tokens + max_drivable_tokens)
        self.category_embedding = nn.Embedding(num_categories, dim)
        self.geometry = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.LayerNorm(dim))
        self.norm = nn.LayerNorm(dim)

    def build_masks(self, objects: list[dict[str, Any]], transform_meta: dict[str, Any], patch_grid: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_size = _image_size(transform_meta)
        masks, cats, geom = [], [], []
        for idx, obj in enumerate(objects[: self.max_tokens]):
            if "box2d" in obj:
                mask = box2d_to_patch_weights(obj["box2d"], image_size=image_size, patch_grid=patch_grid)
                box = obj["box2d"]
                x1, y1, x2, y2 = float(box.get("x1", 0)), float(box.get("y1", 0)), float(box.get("x2", 0)), float(box.get("y2", 0))
            elif "poly2d" in obj:
                mask = poly2d_to_patch_weights(obj["poly2d"], image_size=image_size, patch_grid=patch_grid)
                x1 = y1 = 0.0
                x2, y2 = image_size
            else:
                continue
            w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
            iw, ih = image_size
            geom.append(torch.tensor([x1 / iw, y1 / ih, x2 / iw, y2 / ih, (x1 + x2) / (2 * iw), (y1 + y2) / (2 * ih), w / iw, h / ih, (w * h) / max(1.0, iw * ih)], dtype=torch.float32))
            masks.append(mask)
            cats.append(idx % self.category_embedding.num_embeddings)
        if not masks:
            return torch.zeros(0, patch_grid[0] * patch_grid[1]), torch.zeros(0, dtype=torch.long), torch.zeros(0, 9)
        return torch.stack(masks, 0), torch.tensor(cats, dtype=torch.long), torch.stack(geom, 0)

    def forward(self, patch_tokens: torch.Tensor, roi_masks: torch.Tensor, category_ids: torch.Tensor, geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = masked_average_pool(patch_tokens, roi_masks)
        tokens = pooled.to(dtype=patch_tokens.dtype) + self.category_embedding(category_ids.to(patch_tokens.device)) + self.geometry(geometry.to(patch_tokens.device, dtype=patch_tokens.dtype))
        valid = roi_masks.sum(dim=-1) > 0
        return {"evidence_tokens": self.norm(tokens), "valid_mask": valid}
