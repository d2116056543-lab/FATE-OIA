from __future__ import annotations

import torch
from torch import nn

import utils as dino_utils
import vision_transformer as vits


class ACPRDinoFieldExtractor(nn.Module):
    def __init__(
        self,
        arch: str = "vit_small",
        patch_size: int = 8,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        checkpoint_key: str = "teacher",
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        freeze_backbone: bool = True,
        use_mock_dino: bool = False,
        mock_dim: int = 384,
    ) -> None:
        super().__init__()
        self.selected_layers = tuple(int(x) for x in selected_layers)
        self.patch_size = patch_size
        self.grid_hw = (45, 80)
        self.original_tokens = 3601
        self.use_mock_dino = use_mock_dino
        self.dim = mock_dim
        if use_mock_dino:
            self.backbone = nn.Conv2d(3, mock_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        else:
            self.backbone = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
            dino_utils.load_pretrained_weights(self.backbone, pretrained_weights, checkpoint_key, arch, patch_size)
            self.dim = self.backbone.embed_dim
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone.eval()

    def _mock_forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        with torch.no_grad():
            patch = self.backbone(images).flatten(2).transpose(1, 2)
            patch = patch[:, : self.grid_hw[0] * self.grid_hw[1]]
            cls = patch.mean(1, keepdim=True)
            patches = torch.stack([patch for _ in self.selected_layers], dim=1)
            cls_layers = torch.stack([cls.squeeze(1) for _ in self.selected_layers], dim=1)
        return {
            "patch_tokens_by_layer": patches,
            "cls_tokens_by_layer": cls_layers,
            "patch_tokens_last": patches[:, -1],
            "cls_token_last": cls_layers[:, -1],
            "grid_hw": self.grid_hw,
            "original_tokens": self.original_tokens,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        if images.shape[-2:] != (360, 640):
            raise ValueError(f"ACPR expects 360x640 images, got {tuple(images.shape[-2:])}")
        if self.use_mock_dino:
            return self._mock_forward(images)
        with torch.no_grad():
            x = self.backbone.prepare_tokens(images)
            outputs: list[torch.Tensor] = []
            for idx, block in enumerate(self.backbone.blocks, start=1):
                if idx == 1:
                    x = block(x)
                else:
                    x = block(x)
                if idx in self.selected_layers:
                    outputs.append(self.backbone.norm(x))
            if len(outputs) != len(self.selected_layers):
                raise RuntimeError(f"Requested layers {self.selected_layers}, collected {len(outputs)}")
            cls = torch.stack([o[:, 0] for o in outputs], dim=1)
            patch = torch.stack([o[:, 1:] for o in outputs], dim=1)
        if patch.shape[2] != 3600:
            raise RuntimeError(f"Expected 3600 patch tokens, got {patch.shape[2]}")
        patch = patch.detach()
        cls = cls.detach()
        return {
            "patch_tokens_by_layer": patch,
            "cls_tokens_by_layer": cls,
            "patch_tokens_last": patch[:, -1],
            "cls_token_last": cls[:, -1],
            "grid_hw": self.grid_hw,
            "original_tokens": self.original_tokens,
        }
