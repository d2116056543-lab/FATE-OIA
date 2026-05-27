from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class EvidenceBuildResult:
    tokens: torch.Tensor
    mask: torch.Tensor
    info: list[list[dict[str, Any]]]


class EvidenceTokenBuilder(nn.Module):
    """Build optional structured evidence tokens from patch tokens.

    The fair score branch starts with ``patch_only`` so no GT evidence is used at
    inference. GT modes are intentionally explicit and reported in manifests.
    """

    VALID_MODES = {"patch_only", "train_gt_eval_patch", "gt_evidence_upper_bound", "pseudo_evidence"}

    def __init__(self, dim: int, max_evidence_tokens: int = 32, evidence_mode: str = "patch_only") -> None:
        super().__init__()
        if evidence_mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported evidence_mode: {evidence_mode}")
        self.dim = int(dim)
        self.max_evidence_tokens = int(max_evidence_tokens)
        self.evidence_mode = evidence_mode
        self.category_embed = nn.Embedding(64, dim)
        self.geometry_proj = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.scene_type_id = 0

    @property
    def uses_gt_evidence_at_eval(self) -> bool:
        return self.evidence_mode == "gt_evidence_upper_bound"

    def _empty(self, patch_tokens: torch.Tensor) -> EvidenceBuildResult:
        b, _, d = patch_tokens.shape
        return EvidenceBuildResult(
            tokens=patch_tokens.new_zeros((b, 0, d)),
            mask=torch.zeros((b, 0), dtype=torch.bool, device=patch_tokens.device),
            info=[[] for _ in range(b)],
        )

    def _scene_token(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[list[dict[str, Any]]]]:
        b, n, d = patch_tokens.shape
        pooled = patch_tokens.mean(dim=1, keepdim=True)
        geom = patch_tokens.new_tensor([[0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]]).expand(b, 1, -1)
        token = pooled + self.category_embed(torch.zeros(b, 1, dtype=torch.long, device=patch_tokens.device)) + self.geometry_proj(geom)
        mask = torch.ones((b, 1), dtype=torch.bool, device=patch_tokens.device)
        info = [[{"category": "scene", "source": "scene", "token_index": 0}] for _ in range(b)]
        return token, mask, info

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_grid: tuple[int, int] | list[int] | None = None,
        evidence_metadata: list[dict[str, Any]] | None = None,
        *,
        train: bool = True,
    ) -> EvidenceBuildResult:
        if patch_tokens.ndim != 3:
            raise ValueError(f"patch_tokens must be [B,N,D], got {tuple(patch_tokens.shape)}")
        if self.evidence_mode == "patch_only" or self.evidence_mode == "pseudo_evidence":
            return self._empty(patch_tokens)
        if self.evidence_mode == "train_gt_eval_patch" and not train:
            return self._empty(patch_tokens)
        # Robust minimal GT path: always add a scene evidence token if the run is
        # explicitly marked as GT-evidence. Full object/poly ROI pooling can use
        # evidence_metadata later without changing the public API.
        token, mask, info = self._scene_token(patch_tokens)
        if self.max_evidence_tokens <= 0:
            return self._empty(patch_tokens)
        return EvidenceBuildResult(tokens=token[:, : self.max_evidence_tokens], mask=mask[:, : self.max_evidence_tokens], info=info)
