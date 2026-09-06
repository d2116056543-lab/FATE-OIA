from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

import utils as dino_utils
import vision_transformer as vits


class CoEVVisualField(nn.Module):
    """One DINO instance with a frozen eight-block prefix and trainable upper four blocks."""

    def __init__(self, pretrained_weights: str, patch_size: int = 8,
                 activation_checkpointing: bool = True, use_mock: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.activation_checkpointing = activation_checkpointing
        self.use_mock = use_mock
        if use_mock:
            self.backbone = nn.Module()
            self.backbone.patch_embed = nn.Conv2d(3, 384, patch_size, patch_size)
            self.backbone.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 384)) for _ in range(12)])
            self.backbone.norm = nn.LayerNorm(384)
        else:
            self.backbone = vits.vit_small(patch_size=patch_size, num_classes=0)
            dino_utils.load_pretrained_weights(self.backbone, pretrained_weights, "teacher", "vit_small", patch_size)
        self.measurement_norm = nn.LayerNorm(384)
        self.task_norm = nn.LayerNorm(384)
        if not use_mock:
            self.measurement_norm.load_state_dict(self.backbone.norm.state_dict())
            self.task_norm.load_state_dict(self.backbone.norm.state_dict())
        for p in self.measurement_norm.parameters():
            p.requires_grad = False
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith(tuple(f"blocks.{i}." for i in range(8, 12)))
        # Original output norm is not used, avoiding two registered trainable task norms.
        for p in self.backbone.norm.parameters():
            p.requires_grad = False

    def _prepare(self, rgb: Tensor) -> Tensor:
        if self.use_mock:
            x = self.backbone.patch_embed(rgb).flatten(2).transpose(1, 2)
            return torch.cat((x.mean(1, keepdim=True), x), 1)
        return self.backbone.prepare_tokens(rgb)

    def encode_frame(self, rgb: Tensor) -> dict[str, Tensor | tuple[int, int]]:
        h, w = rgb.shape[-2:]
        if h % self.patch_size or w % self.patch_size:
            raise ValueError("RGB dimensions must be divisible by patch size")
        with torch.no_grad():
            x = self._prepare(rgb)
            frozen = {}
            for index in range(8):
                x = self.backbone.blocks[index](x)
                if index + 1 in (4, 8):
                    frozen[index + 1] = x
            low8 = self.measurement_norm(frozen[8])[:, 1:]
        # The frozen block activations are constants, but the shared task norm is trainable.
        task4 = self.task_norm(frozen[4].detach())[:, 1:]
        task8 = self.task_norm(frozen[8].detach())[:, 1:]
        x = frozen[8].detach()
        for index in range(8, 12):
            block = self.backbone.blocks[index]
            if self.training and self.activation_checkpointing:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        task12 = self.task_norm(x)[:, 1:]
        return {"low8": low8, "task4": task4, "task8": task8, "task12": task12,
                "grid_hw": (h // self.patch_size, w // self.patch_size)}

    @torch.no_grad()
    def encode_measurement_frame(self, rgb: Tensor) -> Tensor:
        x = self._prepare(rgb)
        for index in range(8):
            x = self.backbone.blocks[index](x)
        return self.measurement_norm(x)[:, 1:]

    def encode_clip(self, target_rgb: Tensor, history_rgb: Tensor, chunk_size: int = 1) -> dict[str, Tensor]:
        b, t = history_rgb.shape[:2]
        history = history_rgb.flatten(0, 1)
        chunks = [self.encode_frame(history[start:start + chunk_size])
                  for start in range(0, history.shape[0], chunk_size)]
        target = self.encode_frame(target_rgb)
        result = {}
        for key in ("low8", "task4", "task8", "task12"):
            joined = torch.cat([x[key] for x in chunks], 0)
            result[f"history_{key}"] = joined.view(b, t, *joined.shape[1:])
            result[f"target_{key}"] = target[key]
        result["history_grid_hw"] = chunks[0]["grid_hw"]
        result["target_grid_hw"] = target["grid_hw"]
        return result

    def train(self, mode: bool = True):
        super().train(mode)
        self.measurement_norm.eval()
        return self
