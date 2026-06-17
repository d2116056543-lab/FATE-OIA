from __future__ import annotations


def pmt_phase_for_epoch(epoch: int, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    phase1_end = int(cfg.get("phase1_end_epoch", 2))
    phase2_start = int(cfg.get("phase2_start_epoch", 3))
    stable_start = int(cfg.get("stable_start_epoch", 9))
    if epoch <= phase1_end:
        return {"phase": "warmup", "triadic_enabled": False, "threshold_delta_enabled": False, "pair_cap_ratio": 0.10, "triadic_lr_multiplier": 1.0, "threshold_delta_lr_multiplier": 1.0}
    if epoch >= stable_start:
        return {"phase": "stable", "triadic_enabled": True, "threshold_delta_enabled": True, "pair_cap_ratio": float(cfg.get("stable_pair_cap_ratio", 0.05)), "triadic_lr_multiplier": float(cfg.get("stable_triadic_lr_multiplier", 0.5)), "threshold_delta_lr_multiplier": float(cfg.get("stable_threshold_delta_lr_multiplier", 0.5))}
    return {"phase": "pmt", "triadic_enabled": True, "threshold_delta_enabled": True, "pair_cap_ratio": float(cfg.get("pair_loss_cap_ratio_phase2", 0.10)), "triadic_lr_multiplier": 1.0, "threshold_delta_lr_multiplier": 1.0}
