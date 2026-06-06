from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class DINOIntermediateExtractor(nn.Module):
    def __init__(self, backbone: nn.Module, layer_indices: tuple[int, ...] = (3, 6, 9, 12), patch_hw: tuple[int, int] = (45, 80), dim: int = 384, frozen: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_indices = tuple(layer_indices)
        self.patch_hw = tuple(patch_hw)
        self.dim = int(dim)
        if frozen:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> dict[str, object]:
        raw = self.backbone.get_intermediate_layers(images, n=max(len(self.layer_indices), max(self.layer_indices)))
        if not isinstance(raw, (list, tuple)):
            raise TypeError("backbone.get_intermediate_layers must return list/tuple")
        outputs = list(raw)
        h, w = self.patch_hw
        n_patch = h * w
        tokens_by_layer: dict[int, torch.Tensor] = {}
        maps_by_layer: dict[int, torch.Tensor] = {}
        for pos, layer_idx in enumerate(self.layer_indices):
            src_pos = min(pos, len(outputs) - 1)
            tokens = outputs[src_pos]
            if isinstance(tokens, (list, tuple)):
                tokens = tokens[0]
            if tokens.dim() != 3:
                raise ValueError(f"intermediate layer must be [B,N,D], got {tuple(tokens.shape)}")
            if tokens.shape[1] == n_patch + 1:
                tokens = tokens[:, 1:]
            if tokens.shape[1] != n_patch:
                raise ValueError(f"expected {n_patch} patch tokens, got {tokens.shape[1]}")
            tokens_by_layer[int(layer_idx)] = tokens
            maps_by_layer[int(layer_idx)] = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h, w)
        return {"tokens_by_layer": tokens_by_layer, "maps_by_layer": maps_by_layer, "patch_hw": self.patch_hw, "cls_token": None}


class TinyPatchDINOExtractor(nn.Module):
    """Direct-image smoke extractor with DINO-like multi-layer token outputs."""

    def __init__(self, dim: int = 384, layer_indices: tuple[int, ...] = (3, 6, 9, 12), patch_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.layer_indices = tuple(layer_indices)
        self.patch_hw = tuple(patch_hw)
        self.stem = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=max(1, dim)), nn.GELU(), nn.Conv2d(dim, dim, 1))
            for _ in self.layer_indices
        ])
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, images: torch.Tensor) -> dict[str, object]:
        x = self.stem(images)
        if x.shape[-2:] != self.patch_hw:
            x = F.interpolate(x, size=self.patch_hw, mode="bilinear", align_corners=False)
        maps_by_layer: dict[int, torch.Tensor] = {}
        tokens_by_layer: dict[int, torch.Tensor] = {}
        for idx, block in zip(self.layer_indices, self.blocks):
            x = self.norm(x + 0.1 * block(x))
            maps_by_layer[int(idx)] = x
            tokens_by_layer[int(idx)] = x.flatten(2).transpose(1, 2)
        return {"tokens_by_layer": tokens_by_layer, "maps_by_layer": maps_by_layer, "patch_hw": self.patch_hw, "cls_token": None}


def _load_state_dict_flexible(model: nn.Module, checkpoint_path: str | Path, checkpoint_key: str | None = None) -> dict[str, Any]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state: Any = ckpt
    if checkpoint_key and isinstance(ckpt, dict) and checkpoint_key in ckpt:
        state = ckpt[checkpoint_key]
    elif isinstance(ckpt, dict):
        for key in ("teacher", "student", "model", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if isinstance(state, dict):
        clean = {}
        for k, v in state.items():
            nk = k
            for prefix in ("module.", "backbone.", "teacher.", "student."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
            clean[nk] = v
        missing, unexpected = model.load_state_dict(clean, strict=False)
        return {"missing": list(missing), "unexpected": list(unexpected), "checkpoint_path": str(checkpoint_path)}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def build_dino_extractor(
    arch: str = "vit_small",
    patch_size: int = 8,
    pretrained_weights: str | None = None,
    checkpoint_key: str | None = None,
    layer_indices: tuple[int, ...] = (3, 6, 9, 12),
    patch_hw: tuple[int, int] = (45, 80),
    dim: int = 384,
    frozen: bool = True,
) -> nn.Module:
    """Build real DINO ViT-S/8 when weights are provided, otherwise smoke fallback."""
    if pretrained_weights:
        import vision_transformer as vits

        if arch not in vits.__dict__:
            raise ValueError(f"Unsupported local DINO arch: {arch}")
        backbone = vits.__dict__[arch](patch_size=patch_size)
        load_info = _load_state_dict_flexible(backbone, pretrained_weights, checkpoint_key)
        extractor = DINOIntermediateExtractor(backbone, layer_indices=layer_indices, patch_hw=patch_hw, dim=getattr(backbone, "embed_dim", dim), frozen=frozen)
        extractor.load_info = load_info  # type: ignore[attr-defined]
        return extractor
    return TinyPatchDINOExtractor(dim=dim, layer_indices=layer_indices, patch_hw=patch_hw)
