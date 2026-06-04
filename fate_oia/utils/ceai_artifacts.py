from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def json_safe(obj: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(obj):
            if obj.ndim == 0:
                return obj.detach().cpu().item()
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def make_selected_vs_random_evidence_stats(
    selected_mean: float | None,
    random_mean: float | None,
    *,
    computed: bool,
    margin: float = 0.01,
) -> dict[str, Any]:
    if not computed:
        return {
            "available": False,
            "reason": "not_computed_in_ceai_v1_1",
            "evidence_gate_active": False,
            "selected_mean": None,
            "random_mean": None,
        }
    if selected_mean == 0.0 and random_mean == 0.0:
        return {
            "available": False,
            "reason": "degenerate_zero_zero_not_evidence",
            "evidence_gate_active": False,
            "selected_mean": None,
            "random_mean": None,
        }
    selected = float(selected_mean)
    random = float(random_mean)
    return {
        "available": True,
        "metric_type": "scene_state_proxy_not_deletion",
        "selected_mean": selected,
        "random_mean": random,
        "evidence_gate_active": bool(selected > random + float(margin)),
    }
