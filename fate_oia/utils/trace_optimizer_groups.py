from __future__ import annotations

from typing import Any

import torch


def _lr(args: Any, name: str, default: float) -> float:
    return float(getattr(args, name, default))


def _group_name(name: str) -> str:
    if name == "reason_alpha" or "reason_alpha" in name:
        return "reason_alpha"
    if "action_bias" in name or "safe_ensemble" in name:
        return "action_bias"
    if name.startswith("transport") or name.startswith("evidence_pooler"):
        return "transport"
    if name.startswith("label_corr"):
        return "label_corr"
    if "reason" in name and "reason_to_action" not in name and "action_reason" not in name:
        return "reason_head"
    return "action_head"


def build_action_primary_trace_optimizer(model: torch.nn.Module, args: Any) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "action_head": [],
        "reason_head": [],
        "transport": [],
        "label_corr": [],
        "reason_alpha": [],
        "action_bias": [],
    }
    seen: dict[int, str] = {}
    duplicates: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        gid = _group_name(name)
        if id(param) in seen:
            duplicates.append(name)
            continue
        seen[id(param)] = name
        buckets[gid].append((name, param))
    missing: list[str] = []
    for name, param in model.named_parameters():
        if param.requires_grad and id(param) not in seen:
            missing.append(name)
    specs = [
        ("action_head", _lr(args, "lr_action_head", 3e-4)),
        ("reason_head", _lr(args, "lr_reason_head", 2e-4)),
        ("transport", _lr(args, "lr_transport", 1e-4)),
        ("label_corr", _lr(args, "lr_label_corr", 5e-5)),
        ("reason_alpha", _lr(args, "lr_reason_alpha", 5e-5)),
        ("action_bias", _lr(args, "lr_action_bias", 1e-3)),
    ]
    groups = []
    param_to_group = {}
    for group_name, lr in specs:
        rows = buckets[group_name]
        if not rows:
            continue
        groups.append({"params": [p for _, p in rows], "lr": lr, "group_name": group_name})
        for n, _ in rows:
            param_to_group[n] = group_name
    if "reason_alpha" not in param_to_group:
        raise RuntimeError("reason_alpha is trainable but was not grouped")
    if "action_bias" not in param_to_group:
        raise RuntimeError("action_bias is trainable but was not grouped")
    if missing or duplicates:
        raise RuntimeError(f"optimizer grouping failure missing={missing} duplicates={duplicates}")
    opt = torch.optim.AdamW(groups, weight_decay=float(getattr(args, "weight_decay", 1e-4)))
    report = {
        "groups": [{"name": g["group_name"], "lr": g["lr"], "param_tensors": len(g["params"])} for g in groups],
        "param_to_group": param_to_group,
        "missing_trainable": missing,
        "duplicate_trainable": duplicates,
        "trainable_param_tensors": len(param_to_group),
        "dino_backbone_in_optimizer": False,
    }
    return opt, report
