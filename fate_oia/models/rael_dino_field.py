"""Frozen, cache-free DINO ViT-S/8 dense field for RAEL-OIA.

The extractor intentionally owns only the official DINO visual backbone.  It
does not import an ACPR model, read feature files, or maintain a feature cache.
Callers that need canonical and mirrored images must concatenate them before a
single extractor call.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from torch import nn

import vision_transformer as vits


class RAELDinoFieldExtractor(nn.Module):
    """Expose four frozen DINO ViT-S/8 token fields at 360x640."""

    _EXPECTED_ARCH = "vit_small"
    _EXPECTED_PATCH_SIZE = 8
    _EXPECTED_LAYERS = (3, 6, 9, 12)
    _GRID_HW = (45, 80)
    _ORIGINAL_TOKENS = 3601
    _EMBED_DIM = 384
    _TRANSIENT_REFERENCE_NAMES = frozenset(
        {
            "attention",
            "attention_map",
            "attn_gradients",
            "last_attention",
            "saved_attention",
            "input",
            "inputs",
            "saved_input",
            "v",
            "vproj",
            "value",
            "values",
            "saved_v",
            "weighted_norm",
        }
    )

    def __init__(
        self,
        arch: str = _EXPECTED_ARCH,
        patch_size: int = _EXPECTED_PATCH_SIZE,
        selected_layers: tuple[int, ...] = _EXPECTED_LAYERS,
        checkpoint_key: str = "teacher",
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if arch != self._EXPECTED_ARCH:
            raise ValueError(f"RAEL requires arch={self._EXPECTED_ARCH!r}, got {arch!r}")
        if patch_size != self._EXPECTED_PATCH_SIZE:
            raise ValueError(f"RAEL requires patch_size={self._EXPECTED_PATCH_SIZE}, got {patch_size}")
        if tuple(selected_layers) != self._EXPECTED_LAYERS:
            raise ValueError(
                "RAEL requires selected_layers="
                f"{self._EXPECTED_LAYERS}, got {tuple(selected_layers)}"
            )
        if checkpoint_key != "teacher":
            raise ValueError("RAEL requires checkpoint_key='teacher'")

        self.arch = arch
        self.patch_size = patch_size
        self.selected_layers = tuple(selected_layers)
        self.checkpoint_key = checkpoint_key
        self.pretrained_weights = str(pretrained_weights)
        self.grid_hw = self._GRID_HW
        self.original_tokens = self._ORIGINAL_TOKENS
        # This lifetime diagnostic is process-local.  The lock makes thread
        # sharing explicit; DDP processes each keep their own instance count.
        self.register_buffer(
            "_lifetime_dino_call_count",
            torch.zeros((), dtype=torch.long),
            persistent=False,
        )
        self._dino_call_lock = Lock()
        self.weight_load_report: dict[str, Any] | None = None

        if backbone is None:
            self.backbone = self._build_official_backbone()
            self._load_teacher_checkpoint()
        else:
            self.backbone = backbone

        self.dim = int(getattr(self.backbone, "embed_dim", self._EMBED_DIM))
        if self.dim != self._EMBED_DIM:
            raise RuntimeError(
                f"RAEL DINO ViT-S/8 requires embed_dim={self._EMBED_DIM}, got {self.dim}"
            )
        if len(getattr(self.backbone, "blocks", ())) < self.selected_layers[-1]:
            raise RuntimeError(
                "DINO backbone exposes fewer blocks than the required selected layers "
                f"{self.selected_layers}"
            )
        self._enforce_frozen_eval()

    def _build_official_backbone(self) -> nn.Module:
        try:
            factory = vits.__dict__[self.arch]
        except KeyError as exc:
            raise RuntimeError(f"official DINO architecture is unavailable: {self.arch!r}") from exc
        return factory(patch_size=self.patch_size, num_classes=0)

    def _load_teacher_checkpoint(self) -> None:
        path = Path(self.pretrained_weights)
        if not path.is_file():
            raise FileNotFoundError(
                "RAEL requires an explicit local official DINO ViT-S/8 checkpoint; "
                f"missing {path}"
            )
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Older supported PyTorch releases do not expose weights_only yet.
            checkpoint = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise TypeError("official DINO checkpoint must be a mapping")
        if self.checkpoint_key in checkpoint:
            teacher_state = checkpoint[self.checkpoint_key]
            if not isinstance(teacher_state, Mapping):
                raise TypeError(
                    f"checkpoint[{self.checkpoint_key!r}] must be a state mapping"
                )
            checkpoint_format = "teacher_wrapper"
        elif self._looks_like_direct_backbone_state_dict(checkpoint):
            teacher_state = checkpoint
            checkpoint_format = "raw_backbone_state_dict"
        else:
            raise KeyError(
                "official DINO checkpoint is neither a "
                f"{self.checkpoint_key!r} wrapper nor an explicit raw backbone state dict"
            )

        normalized = {
            self._normalize_checkpoint_key(str(key)): value
            for key, value in teacher_state.items()
        }
        expected = set(self.backbone.state_dict())
        ignored = sorted(key for key in normalized if key.startswith("head."))
        unexpected = sorted(
            key for key in normalized if key not in expected and not key.startswith("head.")
        )
        filtered = {key: value for key, value in normalized.items() if key in expected}
        missing = sorted(expected.difference(filtered))
        if missing or unexpected:
            raise RuntimeError(
                "official DINO teacher checkpoint is incompatible: "
                f"missing={missing}, unexpected={unexpected}"
            )
        result = self.backbone.load_state_dict(filtered, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "official DINO teacher checkpoint did not load strictly: "
                f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
            )
        self.weight_load_report = {
            "path": str(path),
            "checkpoint_key": self.checkpoint_key,
            "checkpoint_format": checkpoint_format,
            "ignored_head_keys": ignored,
            "loaded_keys": len(filtered),
        }

    def _looks_like_direct_backbone_state_dict(self, checkpoint: Mapping[Any, Any]) -> bool:
        """Accept raw weights only when most keys demonstrably target this DINO."""
        normalized = {
            self._normalize_checkpoint_key(str(key)): value
            for key, value in checkpoint.items()
        }
        expected = set(self.backbone.state_dict())
        matching = expected.intersection(normalized)
        # A direct checkpoint must be overwhelmingly identifiable as this
        # backbone before we interpret its missing keys as a DINO mismatch.
        required_matches = max(3, (9 * len(expected) + 9) // 10)
        return (
            len(matching) >= required_matches
            and all(torch.is_tensor(value) for value in normalized.values())
        )

    @staticmethod
    def _normalize_checkpoint_key(key: str) -> str:
        value = key
        for prefix in ("module.", "backbone."):
            while value.startswith(prefix):
                value = value[len(prefix) :]
        return value

    def _enforce_frozen_eval(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True) -> "RAELDinoFieldExtractor":
        """Allow the parent module to train while keeping DINO in eval mode."""
        super().train(mode)
        self._enforce_frozen_eval()
        return self

    @staticmethod
    def concat_canonical_and_mirror(
        canonical_images: torch.Tensor,
        mirror_images: torch.Tensor,
    ) -> torch.Tensor:
        """Build the one-call canonical+mirror batch without invoking DINO."""
        if canonical_images.shape != mirror_images.shape:
            raise ValueError(
                "canonical and mirror batches must have identical shapes, got "
                f"{tuple(canonical_images.shape)} and {tuple(mirror_images.shape)}"
            )
        return torch.cat((canonical_images, mirror_images), dim=0)

    def _validate_images(self, images: torch.Tensor) -> None:
        if images.ndim != 4:
            raise ValueError(f"RAEL DINO expects rank-4 [B,3,360,640], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(f"RAEL DINO expects 3-channel RGB images, got {images.shape[1]} channels")
        if tuple(images.shape[-2:]) != (360, 640):
            raise ValueError(
                "RAEL DINO expects 360x640 images, got "
                f"{tuple(images.shape[-2:])}"
            )

    @staticmethod
    def _contains_tensor_reference(value: Any) -> bool:
        if torch.is_tensor(value):
            return True
        if isinstance(value, (tuple, list)):
            return any(RAELDinoFieldExtractor._contains_tensor_reference(item) for item in value)
        if isinstance(value, dict):
            return any(
                RAELDinoFieldExtractor._contains_tensor_reference(item)
                for item in value.values()
            )
        return False

    def _clear_transient_references(self, root: nn.Module | None = None) -> None:
        """Release tensor references cached by third-party attention blocks."""
        for module in (root or self.backbone).modules():
            for name in self._TRANSIENT_REFERENCE_NAMES:
                if name not in vars(module):
                    continue
                value = getattr(module, name)
                if self._contains_tensor_reference(value):
                    setattr(module, name, None)

    @staticmethod
    def _autocast_context(images: torch.Tensor):
        cuda_bf16_available = (
            images.device.type == "cuda"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        )
        if cuda_bf16_available:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        # CPU and CUDA devices without BF16 support retain the fp32 token contract.
        return nullcontext()

    @property
    def lifetime_dino_call_count(self) -> int:
        with self._dino_call_lock:
            return int(self._lifetime_dino_call_count.detach().cpu().item())

    def _record_dino_call(self) -> int:
        with self._dino_call_lock:
            self._lifetime_dino_call_count.add_(1)
            return int(self._lifetime_dino_call_count.detach().cpu().item())

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int] | int]:
        self._validate_images(images)
        self._enforce_frozen_eval()

        outputs: list[torch.Tensor] = []
        try:
            with torch.no_grad(), self._autocast_context(images):
                tokens = self.backbone.prepare_tokens(images)
                lifetime_dino_call_count = self._record_dino_call()
                if tokens.ndim != 3 or tokens.shape[1] != self.original_tokens:
                    raise RuntimeError(
                        "DINO token contract requires [B,3601,384], got "
                        f"{tuple(tokens.shape)}"
                    )
                if tokens.shape[2] != self._EMBED_DIM:
                    raise RuntimeError(
                        "DINO embedding contract requires 384 channels, got "
                        f"{tokens.shape[2]}"
                    )
                for index, block in enumerate(self.backbone.blocks, start=1):
                    try:
                        tokens = block(tokens)
                        if index in self.selected_layers:
                            normalized = self.backbone.norm(tokens)
                            if normalized.shape != tokens.shape:
                                raise RuntimeError(
                                    "DINO normalization changed token shape from "
                                    f"{tuple(tokens.shape)} to {tuple(normalized.shape)}"
                                )
                            outputs.append(normalized)
                    finally:
                        # DINO Attention stores attention_map/input/v on each
                        # block. Release this block before the next one starts.
                        self._clear_transient_references(block)
                if len(outputs) != len(self.selected_layers):
                    raise RuntimeError(
                        f"requested layers {self.selected_layers}, collected {len(outputs)}"
                    )
                stacked = torch.stack(outputs, dim=1)
                patch_tokens = stacked[:, :, 1:, :].detach()
                cls_tokens = stacked[:, :, 0, :].detach()
            expected_patch_shape = (
                images.shape[0],
                len(self.selected_layers),
                self.grid_hw[0] * self.grid_hw[1],
                self._EMBED_DIM,
            )
            if tuple(patch_tokens.shape) != expected_patch_shape:
                raise RuntimeError(
                    f"DINO patch contract requires {expected_patch_shape}, got "
                    f"{tuple(patch_tokens.shape)}"
                )
            expected_cls_shape = (images.shape[0], len(self.selected_layers), self._EMBED_DIM)
            if tuple(cls_tokens.shape) != expected_cls_shape:
                raise RuntimeError(
                    f"DINO CLS contract requires {expected_cls_shape}, got {tuple(cls_tokens.shape)}"
                )
            return {
                "patch_tokens_by_layer": patch_tokens,
                "cls_tokens_by_layer": cls_tokens,
                "grid_hw": self.grid_hw,
                "original_tokens": self.original_tokens,
                "dino_call_count": 1,
                "lifetime_dino_call_count": lifetime_dino_call_count,
            }
        finally:
            self._clear_transient_references()


__all__ = ["RAELDinoFieldExtractor"]
