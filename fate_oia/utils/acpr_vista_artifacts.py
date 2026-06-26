from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .acpr_artifacts import append_jsonl, json_safe, write_json


def vista_stats_payload(out: dict[str, Any], epoch: int, step: int | None = None) -> dict[str, Any]:
    keys = [
        "vista_alpha_per_layer",
        "vista_alpha_abs_mean",
        "vista_adapter_delta_norm_per_layer",
        "vista_adapter_delta_norm_mean",
        "vista_gate_mean",
        "vista_gate_max",
        "vista_gate_entropy",
        "vista_anchor_mix",
        "vista_base_fraction",
        "vista_learned_fraction",
        "vista_predicate_importance_prior",
        "vista_predicate_importance",
        "vista_delta_mass_on_high_gate",
        "vista_delta_uniformity",
    ]
    payload: dict[str, Any] = {"epoch": int(epoch), "vista_enabled": bool(out.get("vista_enabled", False))}
    if step is not None:
        payload["step"] = int(step)
    for key in keys:
        val = out.get(key)
        if torch.is_tensor(val):
            payload[key] = json_safe(val.detach().cpu())
        elif val is not None:
            payload[key] = val
    return payload


def write_vista_epoch_artifacts(epoch_dir: Path, out: dict[str, Any], epoch: int) -> None:
    append_jsonl(epoch_dir / "vista_diagnostics.jsonl", vista_stats_payload(out, epoch))
    gate = out.get("vista_gate_map")
    if torch.is_tensor(gate):
        torch.save(gate.detach().cpu(), epoch_dir / "vista_gate_map.pt")
    delta = out.get("vista_delta_map")
    if torch.is_tensor(delta):
        torch.save(delta.detach().cpu(), epoch_dir / "vista_delta_map.pt")
    names = out.get("vista_predicate_names")
    importance = out.get("vista_predicate_importance")
    prior = out.get("vista_predicate_importance_prior")
    if isinstance(names, list) and torch.is_tensor(importance):
        rows = []
        imp = importance.detach().cpu().view(-1)
        pri = prior.detach().cpu().view(-1) if torch.is_tensor(prior) else torch.zeros_like(imp)
        for idx, name in enumerate(names[: imp.numel()]):
            rows.append({"predicate_id": idx, "predicate_name": str(name), "importance": float(imp[idx]), "prior": float(pri[idx])})
        write_json(epoch_dir / "predicate_importance_by_name.json", rows)
