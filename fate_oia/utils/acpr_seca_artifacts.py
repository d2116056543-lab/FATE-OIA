from __future__ import annotations

import json
from pathlib import Path

import torch


def _safe_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().mean().cpu()) if x.numel() else 0.0
    try:
        return float(x)
    except Exception:
        return 0.0


def seca_metrics_payload(out: dict, metrics: dict | None = None) -> dict:
    scales = out.get("seca_residual_scale", torch.empty(0))
    legacy = metrics.get("metrics_legacy_base_fixed", {}) if metrics else {}
    seca = metrics.get("metrics_base_fixed", {}) if metrics else {}
    return {
        "available": bool(out.get("seca_enabled", False)),
        "residual_scale": scales.detach().cpu().tolist() if torch.is_tensor(scales) else [],
        "residual_scale_abs_mean": _safe_float(scales.abs() if torch.is_tensor(scales) else 0.0),
        "evidence_context_norm": _safe_float(out.get("seca_evidence_context_norm", 0.0)),
        "null_attention_mean": _safe_float(out.get("seca_null_attention", 0.0)),
        "active_reason_count": _safe_float(out.get("seca_active_reason_count", 0.0)),
        "attention_entropy": _safe_float(out.get("seca_attention_entropy", 0.0)),
        "action_attention_diversity": _safe_float(out.get("seca_action_attention_diversity", 0.0)),
        "legacy_Act_mF1": float(legacy.get("Act_mF1", 0.0) or 0.0),
        "seca_base_Act_mF1": float(seca.get("Act_mF1", 0.0) or 0.0),
        "deploy_Act_mF1": float((metrics or {}).get("metrics_raw_fixed", {}).get("Act_mF1", 0.0) or 0.0),
        "legacy_minus_seca_per_action": [float(a) - float(b) for a, b in zip(legacy.get("per_action_F1", []) or [], seca.get("per_action_F1", []) or [])],
        "seca_minus_legacy_per_action": [float(b) - float(a) for a, b in zip(legacy.get("per_action_F1", []) or [], seca.get("per_action_F1", []) or [])],
    }


def write_seca_case_stub(path: str | Path, *, file_name: str, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"file_name": file_name, **payload}, ensure_ascii=False) + "\n")
