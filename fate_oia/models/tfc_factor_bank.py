from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


ACTION_ALIASES = {"stop_slow": 1, "forward": 0, "turn_left": 2, "turn_right": 3}


@dataclass(frozen=True)
class FactorSpec:
    name: str
    entity: str
    attribute: str
    spatial: str
    polarity: str
    region_prior: str
    action_support: tuple[int, ...]
    action_inhibit: tuple[int, ...]
    reason_support: tuple[int, ...]
    reason_inhibit: tuple[int, ...]
    reason_conflict: tuple[int, ...]
    factor_conflict: tuple[int, ...]


def _resolve_names(names: Any, mapping: dict[str, int], field: str) -> tuple[int, ...]:
    if names is None:
        return ()
    if not isinstance(names, list):
        raise ValueError(f"{field} must be a list")
    out: list[int] = []
    for name in names:
        if isinstance(name, int):
            out.append(int(name))
            continue
        if str(name) not in mapping:
            raise ValueError(f"Unknown {field} target: {name}")
        out.append(mapping[str(name)])
    return tuple(sorted(set(out)))


class TFCFactorBank(nn.Module):
    def __init__(self, specs: list[FactorSpec], action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.specs = specs
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        mats = self._build_matrices()
        for name, tensor in mats.items():
            self.register_buffer(name, tensor)

    @classmethod
    def from_yaml(cls, path: str | Path, action_dim: int = 4, reason_dim: int = 21) -> "TFCFactorBank":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg.get("version") != "acpr_tfc_v1":
            raise ValueError("configs/acpr_tfc_factors.yaml must use version=acpr_tfc_v1")
        reason_aliases = {str(k): int(v) for k, v in (cfg.get("reason_targets", {}).get("aliases") or {}).items()}
        if int(cfg.get("reason_targets", {}).get("count", reason_dim)) != reason_dim:
            raise ValueError("reason_targets.count must match reason_dim")
        factors = cfg.get("factors") or {}
        if not factors:
            raise ValueError("No factors configured")
        names = list(factors.keys())
        name_to_idx = {name: i for i, name in enumerate(names)}
        specs: list[FactorSpec] = []
        for name, raw in factors.items():
            scope = raw.get("target_scope") or {}
            conflicts = list(scope.get("factor_conflict") or [])
            for c in conflicts:
                if c not in name_to_idx:
                    raise ValueError(f"Factor {name} references missing conflict factor {c}")
            specs.append(
                FactorSpec(
                    name=name,
                    entity=str(raw["entity"]),
                    attribute=str(raw["attribute"]),
                    spatial=str(raw["spatial"]),
                    polarity=str(raw["polarity"]),
                    region_prior=str(raw["region_prior"]),
                    action_support=_resolve_names(scope.get("action_support"), ACTION_ALIASES, "action_support"),
                    action_inhibit=_resolve_names(scope.get("action_inhibit"), ACTION_ALIASES, "action_inhibit"),
                    reason_support=_resolve_names(scope.get("reason_support"), reason_aliases, "reason_support"),
                    reason_inhibit=_resolve_names(scope.get("reason_inhibit"), reason_aliases, "reason_inhibit"),
                    reason_conflict=_resolve_names(scope.get("reason_conflict"), reason_aliases, "reason_conflict"),
                    factor_conflict=tuple(name_to_idx[c] for c in conflicts),
                )
            )
        required_mirrors = [
            ("lane_left_available", "lane_right_available"),
            ("left_solid_boundary", "right_solid_boundary"),
            ("lane_absent_left", "lane_absent_right"),
        ]
        for left, right in required_mirrors:
            if (left in name_to_idx) != (right in name_to_idx):
                raise ValueError(f"Incomplete left/right mirror pair: {left}/{right}")
        if "traffic_light_red" in name_to_idx and "traffic_light_green" in name_to_idx:
            red = specs[name_to_idx["traffic_light_red"]]
            green = specs[name_to_idx["traffic_light_green"]]
            if name_to_idx["traffic_light_green"] not in red.factor_conflict or name_to_idx["traffic_light_red"] not in green.factor_conflict:
                raise ValueError("traffic_light_red/green contradiction must be symmetric")
        return cls(specs, action_dim=action_dim, reason_dim=reason_dim)

    @property
    def num_factors(self) -> int:
        return len(self.specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def _build_matrices(self) -> dict[str, torch.Tensor]:
        f = len(self.specs)
        fas = torch.zeros(f, self.action_dim)
        fai = torch.zeros(f, self.action_dim)
        frs = torch.zeros(f, self.reason_dim)
        fri = torch.zeros(f, self.reason_dim)
        conflict = torch.zeros(f, f)
        sim = torch.zeros(f, f)
        region_names: list[str] = []
        for i, spec in enumerate(self.specs):
            for a in spec.action_support:
                fas[i, a] = 1.0
            for a in spec.action_inhibit:
                fai[i, a] = 1.0
            for r in spec.reason_support:
                frs[i, r] = 1.0
            for r in set(spec.reason_inhibit + spec.reason_conflict):
                fri[i, r] = 1.0
            for j in spec.factor_conflict:
                conflict[i, j] = 1.0
                conflict[j, i] = 1.0
            region_names.append(spec.region_prior)
        for i, a in enumerate(self.specs):
            for j, b in enumerate(self.specs):
                if i == j:
                    sim[i, j] = 1.0
                elif conflict[i, j] > 0:
                    sim[i, j] = -1.0
                elif a.entity == b.entity or a.spatial.replace("left", "right") == b.spatial.replace("right", "left"):
                    sim[i, j] = 0.5
        region_to_id = {name: idx for idx, name in enumerate(sorted(set(region_names)))}
        region_id = torch.tensor([region_to_id[name] for name in region_names], dtype=torch.long)
        return {
            "factor_to_action_support": fas,
            "factor_to_action_inhibit": fai,
            "factor_to_reason_support": frs,
            "factor_to_reason_inhibit": fri,
            "factor_conflict": conflict,
            "native_similarity": sim,
            "region_id": region_id,
        }

    def compatibility_matrices(self) -> dict[str, torch.Tensor]:
        return {
            "factor_to_action_support": self.factor_to_action_support,
            "factor_to_action_inhibit": self.factor_to_action_inhibit,
            "factor_to_reason_support": self.factor_to_reason_support,
            "factor_to_reason_inhibit": self.factor_to_reason_inhibit,
            "factor_conflict": self.factor_conflict,
            "native_similarity": self.native_similarity,
            "region_id": self.region_id,
        }
