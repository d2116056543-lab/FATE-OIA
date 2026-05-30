from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fate_oia.datasets.bdd100k_grounding import load_bdd100k_objects
from fate_oia.engine.audit_cafe_evidence_cache import resolve_grounding_record
from fate_oia.grounding.mask_builder import drivable_map_to_mask, objects_to_mask, poly2d_to_mask


@dataclass
class EvidenceConfig:
    dim: int = 384
    max_evidence_units_per_image: int = 96
    per_reason_topk_evidence: int = 8
    patch_grid_h: int = 45
    patch_grid_w: int = 80
    use_dino_topk_fallback: bool = True
    fallback_quality_multiplier: float = 0.20


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _record_path(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str):
        return value
    paths = record.get("paths")
    if isinstance(paths, dict) and isinstance(paths.get(key), str):
        return paths[key]
    return None


def _objects_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    objects = record.get("objects")
    if isinstance(objects, list):
        return [o for o in objects if isinstance(o, dict)]
    label_json = record.get("label_json")
    if isinstance(label_json, dict):
        out: list[dict[str, Any]] = []
        for frame in label_json.get("frames", []) or []:
            if isinstance(frame, dict):
                out.extend([o for o in frame.get("objects", []) or [] if isinstance(o, dict)])
                out.extend([o for o in frame.get("labels", []) or [] if isinstance(o, dict)])
        return out
    path = _record_path(record, "label_json")
    if path and Path(path).exists():
        try:
            return load_bdd100k_objects(path)
        except Exception:
            return []
    return []


def box_to_patch_mask(box: dict[str, Any], grid_h: int, grid_w: int, image_h: int = 720, image_w: int = 1280) -> torch.Tensor:
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


class CausalEvidencePooler(nn.Module):
    """Construct real object/lane/drivable evidence units plus explicit fallback."""

    def __init__(
        self,
        dim: int = 384,
        max_evidence_units_per_image: int = 96,
        per_reason_topk_evidence: int = 8,
        fallback_quality_multiplier: float = 0.20,
        evidence_pooler_version: str = "v2",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_evidence_units_per_image = int(max_evidence_units_per_image)
        self.per_reason_topk_evidence = int(per_reason_topk_evidence)
        self.fallback_quality_multiplier = float(fallback_quality_multiplier)
        self.evidence_pooler_version = evidence_pooler_version
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

    def _pooled_unit(
        self,
        patch_tokens: torch.Tensor,
        mask: torch.Tensor,
        source_type: int,
        geom_vec: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if mask.numel() != patch_tokens.shape[0] or not bool(mask.any()):
            return None
        pooled = patch_tokens[mask].mean(0)
        pooled = pooled + self.type_embed.weight[source_type].to(pooled.dtype)
        if geom_vec is not None:
            pooled = pooled + self.geom(geom_vec.unsqueeze(0)).squeeze(0).to(pooled.dtype)
        return pooled

    def _reason_mask_for_units(
        self,
        metas: list[dict[str, Any]],
        reason_rules: dict[int, set[str]] | None,
        device: torch.device,
        reason_dim: int = 21,
    ) -> torch.Tensor:
        out = torch.zeros(reason_dim, len(metas), dtype=torch.bool, device=device)
        if not reason_rules:
            return out
        for r, cats in reason_rules.items():
            if not (0 <= int(r) < reason_dim):
                continue
            wanted = {str(c).lower() for c in cats}
            for j, meta in enumerate(metas):
                cat = str(meta.get("category", "")).lower()
                src = str(meta.get("source", "")).lower()
                if cat in wanted or any(cat.startswith(w + "/") for w in wanted) or src in wanted:
                    out[int(r), j] = True
        return out

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
        b, n, d = patch_tokens.shape
        grid_h = max(1, image_height // patch_size)
        grid_w = max(1, image_width // patch_size)
        if grid_h * grid_w != n:
            # Some token-compression paths can expose a mismatched CLS/no-CLS layout.
            grid_h = max(1, int(n**0.5))
            grid_w = max(1, n // grid_h)
        file_names: list[str] = []
        if batch is not None:
            raw = batch.get("file_name", [])
            file_names = [str(raw)] if isinstance(raw, str) else [str(x) for x in raw]

        fallback_tokens, fallback_idx = self._fallback(patch_tokens, label_attention, self.per_reason_topk_evidence)
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        qualities: list[torch.Tensor] = []
        source_types: list[torch.Tensor] = []
        reason_masks: list[torch.Tensor] = []
        meta: list[list[dict[str, Any]]] = []
        key_matches: list[dict[str, str]] = []
        counts = {"object": 0, "lane": 0, "drivable": 0, "fallback": 0}

        for i in range(b):
            sample_tokens: list[torch.Tensor] = []
            sample_masks: list[torch.Tensor] = []
            sample_source: list[int] = []
            sample_meta: list[dict[str, Any]] = []
            fn = file_names[i] if i < len(file_names) else ""
            record, matched_key, match_mode = resolve_grounding_record(fn, grounding_cache or {})
            key_matches.append({"file_name": fn, "matched_key": matched_key, "match_mode": match_mode})
            objects = _objects_from_record(record)
            for obj in objects:
                cat = str(obj.get("category", ""))
                box = obj.get("box2d")
                poly = obj.get("poly2d")
                unit_mask: torch.Tensor | None = None
                source_type = 0
                if isinstance(box, dict) and not cat.startswith(("lane/", "area/")):
                    unit_mask = box_to_patch_mask(box, grid_h, grid_w, image_height, image_width)
                    source_type = 0
                    counts["object"] += 1
                elif poly and cat.startswith("lane/"):
                    unit_mask = poly2d_to_mask(poly, image_size=(image_width, image_height), output_size=(grid_h, grid_w)).bool().flatten()
                    source_type = 1
                    counts["lane"] += 1
                elif poly and cat.startswith("area/"):
                    unit_mask = poly2d_to_mask(poly, image_size=(image_width, image_height), output_size=(grid_h, grid_w)).bool().flatten()
                    source_type = 2
                    counts["drivable"] += 1
                if unit_mask is None or unit_mask.numel() != n or not bool(unit_mask.any()):
                    continue
                box_for_geom = box if isinstance(box, dict) else {}
                geom_vec = torch.tensor(
                    [
                        _safe_float(box_for_geom.get("x1")) / max(image_width, 1),
                        _safe_float(box_for_geom.get("y1")) / max(image_height, 1),
                        _safe_float(box_for_geom.get("x2")) / max(image_width, 1),
                        _safe_float(box_for_geom.get("y2")) / max(image_height, 1),
                        max(_safe_float(box_for_geom.get("x2")) - _safe_float(box_for_geom.get("x1")), 0.0) / max(image_width, 1),
                        max(_safe_float(box_for_geom.get("y2")) - _safe_float(box_for_geom.get("y1")), 0.0) / max(image_height, 1),
                    ],
                    device=patch_tokens.device,
                    dtype=patch_tokens.dtype,
                )
                pooled = self._pooled_unit(patch_tokens[i], unit_mask.to(patch_tokens.device), source_type, geom_vec)
                if pooled is None:
                    continue
                sample_tokens.append(pooled)
                sample_masks.append(unit_mask.to(patch_tokens.device))
                sample_source.append(source_type)
                sample_meta.append({"source": ["object_box", "lane_poly", "drivable"][source_type], "category": cat, "box2d": box if isinstance(box, dict) else None})
                if len(sample_tokens) >= self.max_evidence_units_per_image:
                    break

            if record is not None:
                drive_path = _record_path(record, "drivable_map")
                if drive_path and Path(drive_path).exists() and len(sample_tokens) < self.max_evidence_units_per_image:
                    drive_mask = drivable_map_to_mask(drive_path, output_size=(grid_h, grid_w)).bool().flatten()
                    pooled = self._pooled_unit(patch_tokens[i], drive_mask.to(patch_tokens.device), 2, None)
                    if pooled is not None:
                        sample_tokens.append(pooled)
                        sample_masks.append(drive_mask.to(patch_tokens.device))
                        sample_source.append(2)
                        sample_meta.append({"source": "drivable_map", "category": "area/drivable"})
                        counts["drivable"] += 1

            if not sample_tokens:
                for j in range(fallback_tokens.shape[1]):
                    idx = int(fallback_idx[i, j].item())
                    unit_mask = torch.zeros(n, dtype=torch.bool, device=patch_tokens.device)
                    unit_mask[idx] = True
                    sample_tokens.append(fallback_tokens[i, j] + self.type_embed.weight[3].to(fallback_tokens.dtype))
                    sample_masks.append(unit_mask)
                    sample_source.append(3)
                    sample_meta.append({"source": "fallback_attention", "category": "fallback", "token_index": idx})
                counts["fallback"] += len(sample_tokens)

            ev = torch.stack(sample_tokens, 0)
            q = self.quality(ev).squeeze(-1)
            src_t = torch.tensor(sample_source, device=patch_tokens.device, dtype=torch.long)
            q = torch.where(src_t == 3, q * self.fallback_quality_multiplier, q)
            rows.append(ev)
            masks.append(torch.stack(sample_masks, 0))
            qualities.append(q)
            source_types.append(src_t)
            reason_masks.append(self._reason_mask_for_units(sample_meta, reason_rules, patch_tokens.device))
            meta.append(sample_meta)

        max_len = max(x.shape[0] for x in rows)
        evidence_tokens = patch_tokens.new_zeros((b, max_len, d))
        evidence_mask = torch.zeros((b, max_len), device=patch_tokens.device, dtype=torch.bool)
        evidence_quality = patch_tokens.new_zeros((b, max_len))
        evidence_patch_mask = torch.zeros((b, max_len, n), device=patch_tokens.device, dtype=torch.bool)
        evidence_source_type = torch.full((b, max_len), 3, device=patch_tokens.device, dtype=torch.long)
        reason_evidence_mask = torch.zeros((b, 21, max_len), device=patch_tokens.device, dtype=torch.bool)
        for i, ev in enumerate(rows):
            m = ev.shape[0]
            evidence_tokens[i, :m] = ev
            evidence_mask[i, :m] = True
            evidence_quality[i, :m] = qualities[i]
            evidence_patch_mask[i, :m] = masks[i]
            evidence_source_type[i, :m] = source_types[i]
            reason_evidence_mask[i, :, :m] = reason_masks[i][:, :m]
        real_mask = evidence_mask & (evidence_source_type != 3)
        reason_quality = (reason_evidence_mask.float() * evidence_quality.unsqueeze(1)).amax(-1).clamp(0.0, 1.0)
        return {
            "evidence_tokens": evidence_tokens,
            "evidence_mask": evidence_mask,
            "evidence_quality": evidence_quality,
            "evidence_source_type": evidence_source_type,
            "evidence_patch_mask": evidence_patch_mask,
            "reason_evidence_mask": reason_evidence_mask,
            "action_evidence_mask": real_mask.any(1, keepdim=True).expand(-1, 4),
            "real_evidence_mask": real_mask,
            "reason_quality": reason_quality,
            "meta": meta,
            "key_match": key_matches,
            "counts": counts,
        }
