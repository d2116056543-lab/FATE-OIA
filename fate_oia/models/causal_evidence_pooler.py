from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class EvidenceConfig:
    dim: int = 384
    max_evidence_units_per_image: int = 96
    per_reason_topk_evidence: int = 8
    patch_grid_h: int = 45
    patch_grid_w: int = 80
    use_dino_topk_fallback: bool = True


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def box_to_patch_mask(box: dict[str, Any], grid_h: int, grid_w: int, image_h: int = 720, image_w: int = 1280) -> torch.Tensor:
    """Rasterize a BDD100K box2d into the ViT patch grid."""

    x1 = _safe_float(box.get("x1", box.get("x_min", 0.0)))
    y1 = _safe_float(box.get("y1", box.get("y_min", 0.0)))
    x2 = _safe_float(box.get("x2", box.get("x_max", x1)))
    y2 = _safe_float(box.get("y2", box.get("y_max", y1)))
    gx1 = max(0, min(grid_w - 1, int(x1 / max(image_w, 1) * grid_w)))
    gx2 = max(0, min(grid_w, int(torch.ceil(torch.tensor(x2 / max(image_w, 1) * grid_w)).item())))
    gy1 = max(0, min(grid_h - 1, int(y1 / max(image_h, 1) * grid_h)))
    gy2 = max(0, min(grid_h, int(torch.ceil(torch.tensor(y2 / max(image_h, 1) * grid_h)).item())))
    mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    if gx2 > gx1 and gy2 > gy1:
        mask[gy1:gy2, gx1:gx2] = True
    return mask.flatten()


def _iter_objects(record: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    if not record:
        return []
    objects = record.get("objects")
    if isinstance(objects, list):
        return [o for o in objects if isinstance(o, dict)]
    label_json = record.get("label_json")
    if isinstance(label_json, dict):
        frames = label_json.get("frames", [])
        out: list[dict[str, Any]] = []
        for frame in frames if isinstance(frames, list) else []:
            if isinstance(frame, dict) and isinstance(frame.get("objects"), list):
                out.extend([o for o in frame["objects"] if isinstance(o, dict)])
        return out
    return []


class CausalEvidencePooler(nn.Module):
    """Build compact evidence tokens from object boxes or patch-attention fallback.

    The pooler is deliberately conservative: if true object/lane/drivable evidence is
    missing, it returns fallback tokens and lowers evidence_quality instead of
    pretending that scene-level evidence is ground truth.
    """

    def __init__(self, dim: int = 384, max_evidence_units_per_image: int = 96, per_reason_topk_evidence: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.max_evidence_units_per_image = max_evidence_units_per_image
        self.per_reason_topk_evidence = per_reason_topk_evidence
        self.type_embed = nn.Embedding(4, dim)
        self.geom = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.quality = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Sigmoid())

    def _fallback(self, patch_tokens: torch.Tensor, label_attention: torch.Tensor | None, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
        if label_attention is None:
            score = patch_tokens.norm(dim=-1)
        else:
            attn = label_attention
            if attn.dim() == 4:
                attn = attn.mean(1)
            score = attn[..., 1:].mean(1) if attn.shape[-1] == patch_tokens.shape[1] + 1 else attn[..., : patch_tokens.shape[1]].mean(1)
        k = min(max(1, topk), patch_tokens.shape[1])
        idx = torch.topk(score, k=k, dim=1).indices.sort(dim=1).values
        gathered = torch.gather(patch_tokens, 1, idx.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
        return gathered, idx

    def forward(
        self,
        tokens: torch.Tensor,
        original_tokens: torch.Tensor | None = None,
        label_tokens: torch.Tensor | None = None,
        label_attention: torch.Tensor | None = None,
        batch: dict[str, Any] | None = None,
        grounding_cache: dict[str, dict[str, Any]] | None = None,
        image_height: int = 360,
        image_width: int = 640,
        patch_size: int = 8,
        reason_rules: dict[int, set[str]] | None = None,
    ) -> dict[str, Any]:
        source_tokens = original_tokens if original_tokens is not None else tokens
        patch_tokens = source_tokens[:, 1:] if source_tokens.shape[1] > 1 else source_tokens
        b, _, d = patch_tokens.shape
        grid_h = max(1, image_height // patch_size)
        grid_w = max(1, image_width // patch_size)
        file_names: list[str] = []
        if batch is not None:
            raw = batch.get("file_name", [])
            file_names = [str(raw)] if isinstance(raw, str) else [str(x) for x in raw]
        evidence_rows: list[torch.Tensor] = []
        quality_rows: list[torch.Tensor] = []
        counts = {"object": 0, "lane": 0, "drivable": 0, "fallback": 0}
        meta: list[list[dict[str, Any]]] = []
        topk = min(self.per_reason_topk_evidence, patch_tokens.shape[1])
        fallback_tokens, fallback_idx = self._fallback(patch_tokens, label_attention, topk)

        for i in range(b):
            sample_tokens: list[torch.Tensor] = []
            sample_meta: list[dict[str, Any]] = []
            fn = file_names[i] if i < len(file_names) else ""
            record = grounding_cache.get(fn) if grounding_cache else None
            for obj in _iter_objects(record):
                box = obj.get("box2d")
                if not isinstance(box, dict):
                    continue
                mask = box_to_patch_mask(box, grid_h, grid_w, image_height, image_width).to(patch_tokens.device)
                if mask.numel() != patch_tokens.shape[1] or not bool(mask.any()):
                    continue
                pooled = patch_tokens[i, mask].mean(0)
                x1, y1 = _safe_float(box.get("x1")), _safe_float(box.get("y1"))
                x2, y2 = _safe_float(box.get("x2")), _safe_float(box.get("y2"))
                geom = torch.tensor([x1 / max(image_width, 1), y1 / max(image_height, 1), x2 / max(image_width, 1), y2 / max(image_height, 1), max(x2 - x1, 0) / max(image_width, 1), max(y2 - y1, 0) / max(image_height, 1)], device=patch_tokens.device, dtype=patch_tokens.dtype)
                pooled = pooled + self.type_embed.weight[0].to(pooled.dtype) + self.geom(geom.unsqueeze(0)).squeeze(0)
                sample_tokens.append(pooled)
                sample_meta.append({"source": "object", "category": obj.get("category", ""), "box2d": box})
                counts["object"] += 1
                if len(sample_tokens) >= self.max_evidence_units_per_image:
                    break
            if not sample_tokens:
                counts["fallback"] += int(fallback_tokens.shape[1])
                sample_tokens = [fallback_tokens[i, j] + self.type_embed.weight[3].to(fallback_tokens.dtype) for j in range(fallback_tokens.shape[1])]
                sample_meta = [{"source": "fallback_attention", "token_index": int(fallback_idx[i, j].item())} for j in range(fallback_tokens.shape[1])]
            ev = torch.stack(sample_tokens, 0)
            q = self.quality(ev).squeeze(-1)
            if sample_meta and sample_meta[0].get("source") == "fallback_attention":
                q = q * 0.35
            evidence_rows.append(ev)
            quality_rows.append(q)
            meta.append(sample_meta)

        max_len = max(x.shape[0] for x in evidence_rows)
        evidence_tokens = patch_tokens.new_zeros((b, max_len, d))
        evidence_mask = torch.zeros((b, max_len), device=patch_tokens.device, dtype=torch.bool)
        evidence_quality = patch_tokens.new_zeros((b, max_len))
        for i, ev in enumerate(evidence_rows):
            n = ev.shape[0]
            evidence_tokens[i, :n] = ev
            evidence_mask[i, :n] = True
            evidence_quality[i, :n] = quality_rows[i]
        reason_quality = evidence_quality.mean(1, keepdim=True).expand(-1, 21)
        return {
            "evidence_tokens": evidence_tokens,
            "evidence_mask": evidence_mask,
            "evidence_quality": evidence_quality,
            "reason_quality": reason_quality,
            "meta": meta,
            "counts": counts,
        }

