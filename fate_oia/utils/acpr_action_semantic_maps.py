from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import torch
import yaml


@dataclass
class ActionSemanticMaps:
    action_reason_mask: torch.Tensor
    action_predicate_mask: torch.Tensor
    forbidden_r2a_mask: torch.Tensor
    action_names: list[str]
    reason_names: list[str]
    predicate_names: list[str]
    warnings: list[str]


def _load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _lookup_table(raw: dict, count: int, field: str) -> list[dict]:
    table = raw.get(field, {})
    out = []
    for i in range(count):
        item = table.get(i, table.get(str(i)))
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{i}] missing or invalid")
        out.append(item)
    return out


def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    sums = x.sum(dim=1, keepdim=True)
    return torch.where(sums > 0, x / sums.clamp_min(1e-6), x)


def load_action_semantic_maps(
    grammar_path: str,
    scene_config_path: str,
    action_dim: int = 4,
    reason_dim: int = 21,
) -> ActionSemanticMaps:
    grammar = _load_yaml(grammar_path)
    scene = _load_yaml(scene_config_path)
    actions = _lookup_table(grammar, action_dim, "actions")
    reasons = _lookup_table(grammar, reason_dim, "reasons")
    predicate_rows = scene.get("predicates", [])
    predicate_names = [str(p.get("name", f"predicate_{i}")) for i, p in enumerate(predicate_rows)]
    action_names = [str(a.get("name", f"action_{i}")) for i, a in enumerate(actions)]
    reason_names = [str(r.get("name", "")) for r in reasons]
    if len(action_names) != 4 or len(reason_names) != 21:
        raise ValueError("ACPR FusionLite requires exactly 4 actions and 21 reasons")
    if any(re.fullmatch(r"reason_?\d+", name.strip().lower()) for name in reason_names):
        raise ValueError("Placeholder reason names are forbidden in FusionLite semantic maps")

    display_path = Path(grammar_path).with_name("bdd_oia_reason_names_external.yaml")
    if display_path.exists():
        display = _load_yaml(display_path).get("names", {})
        for i, name in enumerate(reason_names):
            ext = display.get(i, display.get(str(i)))
            if ext is not None and str(ext) != name:
                raise ValueError(f"Reason name mismatch at {i}: grammar={name!r} external={ext!r}")

    action_to_id = {name: i for i, name in enumerate(action_names)}
    pred_to_id = {name: i for i, name in enumerate(predicate_names)}
    action_reason = torch.zeros(action_dim, reason_dim, dtype=torch.float32)
    action_pred = torch.zeros(action_dim, len(predicate_names), dtype=torch.float32)
    forbidden = torch.zeros(action_dim, reason_dim, dtype=torch.float32)
    warnings: list[str] = []
    for rid, reason in enumerate(reasons):
        compatible = [str(x) for x in reason.get("compatible_actions", [])]
        incompatible = [str(x) for x in reason.get("incompatible_actions", [])]
        neutral = set(str(x) for x in reason.get("neutral_actions", []))
        positives = [str(x) for x in reason.get("positive_predicates", [])]
        contradictory = [str(x) for x in reason.get("contradictory_predicates", [])]
        for aname in compatible:
            if aname in action_to_id:
                action_reason[action_to_id[aname], rid] = 1.0
                for pname in positives:
                    if pname in pred_to_id:
                        action_pred[action_to_id[aname], pred_to_id[pname]] += 1.0
                for pname in contradictory:
                    if pname in pred_to_id:
                        action_pred[action_to_id[aname], pred_to_id[pname]] -= 0.25
        for aid, aname in enumerate(action_names):
            if aname in incompatible or (compatible and aname not in compatible and aname not in neutral):
                forbidden[aid, rid] = 1.0
    action_pred = action_pred.clamp_min(0.0)
    action_reason = _normalize_rows(action_reason)
    action_pred = _normalize_rows(action_pred) if action_pred.numel() else action_pred
    for aid, aname in enumerate(action_names):
        if float(action_reason[aid].sum()) <= 0:
            warnings.append(f"empty action_reason_mask for {aname}")
        if action_pred.numel() and float(action_pred[aid].sum()) <= 0:
            warnings.append(f"empty action_predicate_mask for {aname}")
    return ActionSemanticMaps(
        action_reason_mask=action_reason,
        action_predicate_mask=action_pred,
        forbidden_r2a_mask=forbidden,
        action_names=action_names,
        reason_names=reason_names,
        predicate_names=predicate_names,
        warnings=warnings,
    )
