from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


POSITIVE, COUNTER, UNKNOWN = 0, 1, 2


@dataclass(frozen=True)
class StructuredEvidence:
    state_target: torch.Tensor
    state_mask: torch.Tensor
    map_target: torch.Tensor
    map_mask: torch.Tensor
    source_reliability: torch.Tensor
    source_id: torch.Tensor
    source_complete: torch.Tensor
    coverage: dict[str, int]


class LENSStructuredEvidenceBuilder:
    """Fail closed: lack of an explicit, complete source always remains unknown."""

    def __init__(self, schema_path: str | Path, grid_hw: tuple[int, int] = (45, 80)) -> None:
        raw = yaml.safe_load(Path(schema_path).read_text(encoding="utf-8")) or {}
        self.schema = {int(k): v for k, v in (raw.get("reasons") or {}).items()}
        if len(self.schema) != 21:
            raise ValueError("LENS schema must define all 21 reasons")
        self.grid_hw = grid_hw

    def build(self, records: list[dict[str, Any]]) -> StructuredEvidence:
        b, r, n = len(records), 21, self.grid_hw[0] * self.grid_hw[1]
        state = torch.zeros(b, r, 3)
        state[..., UNKNOWN] = 1.0
        state_mask = torch.zeros(b, r)
        map_target = torch.zeros(b, r, n)
        map_mask = torch.zeros(b, r)
        reliability = torch.zeros(b, r)
        source_id = torch.full((b, r), -1, dtype=torch.long)
        complete = torch.zeros(b, r, dtype=torch.bool)
        for bi, record in enumerate(records):
            attributes = record.get("explicit_attributes", {})
            sources = record.get("complete_sources", {})
            for ri, spec in self.schema.items():
                support = any(bool(attributes.get(name, False)) for name in spec.get("support_sources", []))
                counter = any(bool(attributes.get(name, False)) for name in spec.get("counter_sources", []))
                needs_complete = bool(spec.get("complete_source_required", True))
                is_complete = bool(sources.get(spec.get("default_region", ""), False))
                complete[bi, ri] = is_complete
                if support and not counter:
                    state[bi, ri] = torch.tensor([1.0, 0.0, 0.0])
                    state_mask[bi, ri] = 1.0; reliability[bi, ri] = 1.0; source_id[bi, ri] = 1
                elif counter and (is_complete or not needs_complete):
                    state[bi, ri] = torch.tensor([0.0, 1.0, 0.0])
                    state_mask[bi, ri] = 1.0; reliability[bi, ri] = 0.9; source_id[bi, ri] = 2
                # Unknown is deliberately not made a hard target.
        return StructuredEvidence(state, state_mask, map_target, map_mask, reliability, source_id, complete, {
            "known": int(state_mask.sum().item()), "unknown": int((1 - state_mask).sum().item())
        })
