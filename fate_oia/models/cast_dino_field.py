from __future__ import annotations

import torch
from torch import nn

import utils
import vision_transformer as vits


class CastDinoFieldExtractor(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        patch_size: int = 8,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        checkpoint_key: str = "teacher",
        selected_layers: tuple[int, ...] = (3, 7, 11),
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.patch_size = int(patch_size)
        self.pretrained_weights = str(pretrained_weights)
        self.checkpoint_key = checkpoint_key
        self.selected_layers = tuple(int(x) for x in selected_layers)
        self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        utils.load_pretrained_weights(self.backbone, self.pretrained_weights, checkpoint_key, arch, patch_size)
        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @property
    def embed_dim(self) -> int:
        return int(self.backbone.embed_dim)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        h, w = int(images.shape[-2]), int(images.shape[-1])
        grid_hw = (h // self.patch_size, w // self.patch_size)
        x = self.backbone.prepare_tokens(images)
        outputs = []
        for idx, blk in enumerate(self.backbone.blocks):
            x = blk(x)
            if idx in self.selected_layers:
                outputs.append(self.backbone.norm(x))
        if len(outputs) != len(self.selected_layers):
            raise RuntimeError(f"requested layers {self.selected_layers}, collected {len(outputs)}")
        stacked = torch.stack(outputs, dim=1)
        return {
            "cls_tokens_by_layer": stacked[:, :, 0],
            "patch_tokens_by_layer": stacked[:, :, 1:],
            "grid_hw": grid_hw,
            "original_tokens": stacked.shape[2],
        }
