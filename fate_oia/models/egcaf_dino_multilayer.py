from __future__ import annotations

from typing import Any

import torch
from torch import nn

import utils
import vision_transformer as vits


class EGCafDinoMultiLayerExtractor(nn.Module):
    """Direct-image frozen DINO/SNNA ViT-S/8 multi-layer token extractor.

    This module never writes feature caches. For 360x640 and patch 8, patch grid is 45x80.
    """

    def __init__(
        self,
        arch: str = "vit_small",
        patch_size: int = 8,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        checkpoint_key: str = "teacher",
        hook_layers: list[int] | tuple[int, ...] = (3, 6, 9, 12),
        frozen: bool = True,
        lightweight: bool = False,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.hook_layers = [int(x) for x in hook_layers]
        self.patch_size = int(patch_size)
        self.lightweight = bool(lightweight)
        self.embed_dim = int(embed_dim)
        if self.lightweight:
            self.proxy = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            if arch not in vits.__dict__:
                raise ValueError(f"Unsupported DINO arch: {arch}")
            self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            utils.load_pretrained_weights(self.backbone, pretrained_weights, checkpoint_key, arch, patch_size)
            self.embed_dim = int(self.backbone.embed_dim)
            if frozen:
                self.backbone.eval()
                for p in self.backbone.parameters():
                    p.requires_grad = False

    def forward(self, images: torch.Tensor) -> dict[str, Any]:
        _, _, h, w = images.shape
        gh, gw = h // self.patch_size, w // self.patch_size
        if self.lightweight:
            x = self.norm(self.proxy(images).flatten(2).transpose(1, 2))
            layers = {f"layer_{l}": x for l in self.hook_layers}
            cls = {k: v.mean(1) for k, v in layers.items()}
            return {"layer_tokens": layers, "cls_tokens": cls, "grid_hw": (gh, gw), "image_size": (h, w)}
        max_layer = max(self.hook_layers)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            outs = self.backbone.get_intermediate_layers(images, n=max_layer)
        layers: dict[str, torch.Tensor] = {}
        cls: dict[str, torch.Tensor] = {}
        for l in self.hook_layers:
            out = outs[l - 1]
            if out.shape[1] == gh * gw + 1:
                cls[f"layer_{l}"] = out[:, 0]
                tok = out[:, 1:]
            else:
                tok = out
                cls[f"layer_{l}"] = tok.mean(1)
            layers[f"layer_{l}"] = tok
        return {"layer_tokens": layers, "cls_tokens": cls, "grid_hw": (gh, gw), "image_size": (h, w)}
