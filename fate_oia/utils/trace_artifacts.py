from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    try:
        import torch
        if torch.is_tensor(value):
            return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def required_epoch_artifacts() -> list[str]:
    return ["metrics_summary.json", "metrics_raw_fixed.json", "branch_metrics.json", "per_label_reason_metrics.json", "tail_group_metrics.json", "loss_components.jsonl", "transport_stats.json", "prototype_stats.json", "evidence_stats.jsonl", "counterfactual_stats.json", "visual_branch_stats.json", "efficiency_stats.json", "calibration_params_test_diagnostic.json", "failure_cases.jsonl", "trace_visuals_index.jsonl"]
