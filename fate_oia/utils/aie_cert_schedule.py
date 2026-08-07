from __future__ import annotations

import math


def _ramp(progress: float, start: float, end: float) -> float:
    return max(0.0, min(1.0, (progress - start) / max(end - start, 1e-12)))


def schedule_values(optimizer_update: int, schedule_total_updates: int, cfg: dict) -> dict[str, float]:
    p = optimizer_update / max(schedule_total_updates, 1)
    warmup = _ramp(p, 0.0, 0.05)
    cosine_p = _ramp(p, 0.05, 1.0)
    lr = warmup if p < 0.05 else 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * cosine_p))
    return {"lr_multiplier": max(lr, 1e-3), "grounding_scale": 0.25 + 0.75 * warmup,
        "predicate_prior_scale": warmup, "action_scale": 0.10 + 0.90 * _ramp(p, 0.0, 0.10),
        "reason_budget_max": 0.10 + 0.50 * _ramp(p, 0.0, 0.10),
        "transport_gamma_cap": 0.05 + 0.20 * _ramp(p, 0.0, 0.10),
        "cf_scale": _ramp(p, 0.05, 0.15), "ecpo_scale": _ramp(p, 0.05, 0.15),
        "dual_scale": _ramp(p, 0.08, 0.20), "naming_scale": _ramp(p, 0.10, 0.20)}
