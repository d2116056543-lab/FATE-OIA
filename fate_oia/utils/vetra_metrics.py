from __future__ import annotations

import torch


def route_statistics(output: dict) -> dict:
    rows = {}
    for role in ("support", "counter"):
        route = output[f"{role}_route"].detach().float()
        meta = output[f"{role}_meta"]
        named = route[..., :meta["named_count"]].sum(-1)
        unnamed = route[..., meta["named_count"]:meta["named_count"] + meta["unnamed_count"]].sum(-1)
        entropy = -(route.clamp_min(1e-9) * route.clamp_min(1e-9).log()).sum(-1)
        rows.update({f"{role}_null_rate": float(route[..., -1].mean()),
                     f"{role}_named_mass": float(named.mean()),
                     f"{role}_unnamed_mass": float(unnamed.mean()),
                     f"{role}_entropy": float(entropy.mean()),
                     f"{role}_non_null_action_count": int(((1-route[..., -1]).mean(0) > 1e-5).sum())})
    delta = output["vetra_action_delta"].detach().float()
    rows.update({"delta_mean": float(delta.mean()), "delta_std": float(delta.std(unbiased=False)),
                 "delta_abs_max": float(delta.abs().max()),
                 "correction_mean_std_ratio": [float(delta[:, a].mean().abs() / delta[:, a].std(unbiased=False).clamp_min(1e-8)) for a in range(4)],
                 "reason_identity_max_abs": float(output["reason_identity_max_abs"])})
    return rows
