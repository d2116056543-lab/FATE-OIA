from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def save_logits_artifacts(epoch_dir: Path, outputs: dict[str, torch.Tensor], labels: torch.Tensor, file_names: list[str], action_dim: int) -> None:
    logits_dir = epoch_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "action_base_test.pt": outputs["base_action_logits"],
        "action_specialist_test.pt": outputs["action_specialist_logits"],
        "action_final_test.pt": outputs["final_action_logits"],
        "reason_base_test.pt": outputs["base_reason_logits"],
        "reason_specialist_test.pt": outputs["reason_specialist_logits"],
        "reason_final_test.pt": outputs["final_reason_logits"],
        "action_set_test.pt": outputs["action_set_logits"],
    }
    for name, tensor in mapping.items():
        torch.save(tensor.cpu(), logits_dir / name)
    torch.save(labels[:, :action_dim].cpu(), logits_dir / "labels_action_test.pt")
    torch.save(labels[:, action_dim:].cpu(), logits_dir / "labels_reason_test.pt")
    write_json(logits_dir / "file_names_test.json", file_names)


def summarize_scalar_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not rows:
        return out
    keys = sorted({key for row in rows for key in row if isinstance(row.get(key), (int, float))})
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            out[key] = sum(values) / len(values)
    return out
