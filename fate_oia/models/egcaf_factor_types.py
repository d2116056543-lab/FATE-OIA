from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

FACTOR_TYPE_NAMES = [
    "unknown", "front_center", "lower_drivable", "left_lane", "right_lane",
    "traffic_control", "obstacle_object", "vulnerable_user", "global_context",
    "lane_boundary", "tail_or_ambiguous",
]
SOURCE_NAMES = {0: "anchor", 1: "dino_object", 2: "scene_proxy", 3: "global"}


@dataclass
class FactorBatch:
    embeddings: torch.Tensor
    region_masks: torch.Tensor
    boxes: torch.Tensor
    source_ids: torch.Tensor
    type_logits: torch.Tensor
    reliability_init: torch.Tensor
    valid_mask: torch.Tensor
    metadata: dict[str, Any] | None = None
    action_ids: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> "FactorBatch":
        return FactorBatch(
            self.embeddings.to(device),
            self.region_masks.to(device),
            self.boxes.to(device),
            self.source_ids.to(device),
            self.type_logits.to(device),
            self.reliability_init.to(device),
            self.valid_mask.to(device),
            self.metadata or {},
            self.action_ids.to(device) if self.action_ids is not None else None,
        )


def _action_ids_or_default(batch: FactorBatch) -> torch.Tensor:
    if batch.action_ids is not None:
        return batch.action_ids
    return torch.full_like(batch.source_ids, -1)


def concatenate_factor_batches(batches: list[FactorBatch]) -> FactorBatch:
    if not batches:
        raise ValueError("No factor batches")
    return FactorBatch(
        torch.cat([b.embeddings for b in batches], 1),
        torch.cat([b.region_masks for b in batches], 1),
        torch.cat([b.boxes for b in batches], 1),
        torch.cat([b.source_ids for b in batches], 1),
        torch.cat([b.type_logits for b in batches], 1),
        torch.cat([b.reliability_init for b in batches], 1),
        torch.cat([b.valid_mask for b in batches], 1),
        {"sources": [b.metadata for b in batches]},
        torch.cat([_action_ids_or_default(b) for b in batches], 1),
    )


def gather_factors_by_indices(factors: FactorBatch, indices: torch.Tensor) -> FactorBatch:
    b = factors.embeddings.shape[0]
    flat = indices.reshape(b, -1).clamp_min(0)
    d = factors.embeddings.shape[-1]
    emb = torch.gather(factors.embeddings, 1, flat.unsqueeze(-1).expand(-1, -1, d)).reshape(*indices.shape, d)
    h, w = factors.region_masks.shape[-2:]
    masks = torch.gather(
        factors.region_masks, 1, flat.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
    ).reshape(*indices.shape, h, w)
    boxes = torch.gather(factors.boxes, 1, flat.unsqueeze(-1).expand(-1, -1, 4)).reshape(*indices.shape, 4)
    src = torch.gather(factors.source_ids, 1, flat).reshape(indices.shape)
    t = factors.type_logits.shape[-1]
    typ = torch.gather(factors.type_logits, 1, flat.unsqueeze(-1).expand(-1, -1, t)).reshape(*indices.shape, t)
    rel = torch.gather(factors.reliability_init, 1, flat).reshape(indices.shape)
    valid = torch.gather(factors.valid_mask, 1, flat).reshape(indices.shape)
    action_ids = None
    if factors.action_ids is not None:
        action_ids = torch.gather(factors.action_ids, 1, flat).reshape(indices.shape)
    return FactorBatch(emb, masks, boxes, src, typ, rel, valid, {"gather_indices_shape": list(indices.shape)}, action_ids)


def mask_factors(factors: FactorBatch, keep_mask: torch.Tensor) -> FactorBatch:
    keep = keep_mask.float()
    return replace(
        factors,
        embeddings=factors.embeddings * keep.unsqueeze(-1),
        region_masks=factors.region_masks * keep.unsqueeze(-1).unsqueeze(-1),
        reliability_init=factors.reliability_init * keep,
        valid_mask=factors.valid_mask & keep_mask.bool(),
    )


def factor_to_json_records(
    factors: FactorBatch,
    selected_indices: torch.Tensor | None = None,
    selected_weights: torch.Tensor | None = None,
    max_samples: int = 64,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    bsz = min(int(factors.embeddings.shape[0]), max_samples)
    type_ids = factors.type_logits.argmax(-1).detach().cpu()
    action_ids = factors.action_ids.detach().cpu() if factors.action_ids is not None else None
    for b in range(bsz):
        if selected_indices is None:
            for i in range(int(factors.embeddings.shape[1])):
                records.append({
                    "sample_index": b,
                    "factor_index": int(i),
                    "source": SOURCE_NAMES.get(int(factors.source_ids[b, i].item()), "unknown"),
                    "type": FACTOR_TYPE_NAMES[int(type_ids[b, i].item())],
                    "action_conditioned_id": int(action_ids[b, i].item()) if action_ids is not None else -1,
                    "box": [float(x) for x in factors.boxes[b, i].detach().cpu()],
                    "valid": bool(factors.valid_mask[b, i].item()),
                    "weight": None,
                })
        else:
            for a in range(selected_indices.shape[1]):
                for k in range(selected_indices.shape[2]):
                    i = int(selected_indices[b, a, k].item())
                    records.append({
                        "sample_index": b,
                        "action_index": a,
                        "rank": k,
                        "factor_index": i,
                        "source": SOURCE_NAMES.get(int(factors.source_ids[b, i].item()), "unknown"),
                        "type": FACTOR_TYPE_NAMES[int(type_ids[b, i].item())],
                        "action_conditioned_id": int(action_ids[b, i].item()) if action_ids is not None else -1,
                        "box": [float(x) for x in factors.boxes[b, i].detach().cpu()],
                        "valid": bool(factors.valid_mask[b, i].item()),
                        "weight": float(selected_weights[b, a, k].detach().cpu()) if selected_weights is not None else None,
                    })
    return records

