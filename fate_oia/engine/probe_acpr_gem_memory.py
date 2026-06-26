from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config


def candidate_pairs(items: list[str]) -> list[tuple[int, int]]:
    out = []
    for item in items:
        b, a = item.split(":")
        out.append((int(b), int(a)))
    return out


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(cfg: dict, batch_size: int, device: torch.device, gem_enabled: bool) -> dict:
    probe_cfg = dict(cfg)
    probe_cfg["model"] = dict(cfg.get("model", {}))
    probe_cfg["gem"] = dict(cfg.get("gem", {}))
    probe_cfg["gem"]["enabled"] = bool(gem_enabled)
    model = build_model(probe_cfg, device)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    images = torch.randn(batch_size, 3, int(cfg.get("image_height", 360)), int(cfg.get("image_width", 640)), device=device)
    action = torch.rand(batch_size, 4, device=device)
    reason = torch.rand(batch_size, 21, device=device)
    _cuda_sync(device)
    start = time.perf_counter()
    out = model(images, epoch=0)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(out["action_logits_base"], action)
        + torch.nn.functional.binary_cross_entropy_with_logits(out["reason_logits_base"], reason)
    )
    loss.backward()
    _cuda_sync(device)
    elapsed = time.perf_counter() - start
    peak_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else 0.0
    return {
        "batch_size": batch_size,
        "gem_enabled": bool(gem_enabled),
        "step_seconds": elapsed,
        "peak_allocated_gb": peak_gb,
        "loss": float(loss.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", nargs="+", default=["6:5", "5:6", "4:8", "3:10", "2:15"])
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    max_gb = float(cfg.get("runtime", {}).get("target_own_gpu_memory_gb", 30.0))
    results = []
    selected: tuple[int, int] | None = None
    for batch_size, accum in candidate_pairs(args.candidates):
        try:
            base = _measure(cfg, batch_size, device, gem_enabled=False)
            gem = _measure(cfg, batch_size, device, gem_enabled=True)
            overhead = (gem["step_seconds"] / max(base["step_seconds"], 1e-9)) - 1.0
            mem_increase = gem["peak_allocated_gb"] - base["peak_allocated_gb"]
            row = {
                "batch_size": batch_size,
                "gradient_accumulation_steps": accum,
                "effective_batch": batch_size * accum,
                "base": base,
                "gem": gem,
                "forward_backward_overhead": overhead,
                "peak_memory_increase_gb": mem_increase,
                "oom": False,
                "stable": bool(gem["peak_allocated_gb"] <= max_gb and overhead <= 0.75),
            }
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            row = {
                "batch_size": batch_size,
                "gradient_accumulation_steps": accum,
                "effective_batch": batch_size * accum,
                "oom": True,
                "error": str(exc).splitlines()[0],
                "stable": False,
            }
        results.append(row)
        if selected is None and row.get("stable"):
            selected = (batch_size, accum)
    payload = {
        "pass": selected is not None,
        "candidates": candidate_pairs(args.candidates),
        "results": results,
        "selected": selected,
        "device": str(device),
        "max_own_gpu_memory_gb": max_gb,
    }
    (out / "GEM_MEMORY_PASS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
