from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")


def save_epoch_tensors(
    output_dir: str | Path,
    split: str,
    action_logits: torch.Tensor,
    exp_logits: torch.Tensor,
    action_labels: torch.Tensor,
    exp_labels: torch.Tensor,
    file_names: list[str],
    extra_tensors: dict[str, torch.Tensor] | None = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(action_logits.detach().cpu(), out / f"logits_action_{split}.pt")
    torch.save(exp_logits.detach().cpu(), out / f"logits_exp29_{split}.pt")
    torch.save(action_labels.detach().cpu(), out / f"labels_action_{split}.pt")
    torch.save(exp_labels.detach().cpu(), out / f"labels_exp29_{split}.pt")
    write_json(out / f"file_names_{split}.json", file_names)
    for name, tensor in (extra_tensors or {}).items():
        torch.save(tensor.detach().cpu(), out / f"{name}_{split}.pt")
