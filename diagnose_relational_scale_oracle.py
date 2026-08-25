from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F


ROOT = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")


def f1_per_label(logits: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits) >= threshold
    truth = target > 0.5
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def diagnose(prefix: str, count: int) -> dict:
    pre = torch.load(ROOT / f"pre_relational_{prefix}_test.pt", map_location="cpu").float()
    delta = torch.load(ROOT / f"relational_{prefix}_delta_test.pt", map_location="cpu").float()
    final = torch.load(ROOT / f"video_{prefix}_test.pt", map_location="cpu").float()
    target = torch.load(ROOT / f"{prefix}_target_test.pt", map_location="cpu").float()
    threshold = torch.tensor(json.loads((ROOT / "calibration.json").read_text())["image"][:count])
    scales = torch.cat((torch.linspace(0, 8, 33), torch.tensor([12.0, 16.0, 24.0, 32.0, 48.0, 64.0])))
    base_f1 = f1_per_label(pre, target, threshold)
    candidates = torch.stack([pre + scale * delta for scale in scales])
    f1 = torch.stack([f1_per_label(candidate, target, threshold) for candidate in candidates])
    bce = torch.stack([
        F.binary_cross_entropy_with_logits(candidate, target, reduction="none").mean(0)
        for candidate in candidates
    ])
    best_f1_idx = f1.argmax(0)
    best_bce_idx = bce.argmin(0)
    best_f1_logits = torch.stack([
        candidates[best_f1_idx[label], :, label] for label in range(count)
    ], dim=1)
    best_bce_logits = torch.stack([
        candidates[best_bce_idx[label], :, label] for label in range(count)
    ], dim=1)
    return {
        "final_reconstruction_max_abs": float((pre + delta - final).abs().max()),
        "base_mf1": float(base_f1.mean()),
        "scale1_mf1": float(f1_per_label(pre + delta, target, threshold).mean()),
        "oracle_f1_mf1": float(f1_per_label(best_f1_logits, target, threshold).mean()),
        "oracle_f1_scales": scales[best_f1_idx].tolist(),
        "oracle_bce_mf1": float(f1_per_label(best_bce_logits, target, threshold).mean()),
        "oracle_bce_scales": scales[best_bce_idx].tolist(),
        "base_bce": float(F.binary_cross_entropy_with_logits(pre, target)),
        "scale1_bce": float(F.binary_cross_entropy_with_logits(pre + delta, target)),
        "oracle_bce": float(F.binary_cross_entropy_with_logits(best_bce_logits, target)),
    }


print(json.dumps({"action": diagnose("action", 4), "reason": diagnose("reason", 21)}, indent=2))
