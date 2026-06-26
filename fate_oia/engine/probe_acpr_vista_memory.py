from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from fate_oia.engine.train_acpr_oia import build_model, load_config, make_loader


def _one_probe(cfg: dict, batch_size: int, accum: int, device: torch.device) -> dict:
    if device.type != "cuda":
        return {
            "batch_size": batch_size,
            "gradient_accumulation_steps": accum,
            "pass": False,
            "reason": "cuda_required_for_memory_probe",
        }
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cfg = dict(cfg)
    data_cfg = dict(cfg.get("data", {}))
    data_cfg["persistent_workers"] = False
    data_cfg["prefetch_factor"] = 2
    cfg["data"] = data_cfg
    model = build_model(cfg, device).train()
    loader = make_loader(cfg, "train", batch_size, max_samples=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    action = batch["action"].float().to(device, non_blocking=True)
    reason = batch["reason"].float().to(device, non_blocking=True)
    out = model(images, epoch=0)
    loss = (
        torch.nn.functional.binary_cross_entropy_with_logits(out["action_logits_final_raw"], action)
        + torch.nn.functional.binary_cross_entropy_with_logits(out["reason_logits_final_raw"], reason)
        + out["predicate_probs"].mean() * 0.01
        + out["vista_adapter_delta_norm_mean"] * 0.01
    )
    (loss / max(1, accum)).backward()
    allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    del model, loader, batch, images, action, reason, out, loss
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch": batch_size * accum,
        "pass": True,
        "peak_allocated_gb": allocated,
        "peak_reserved_gb": reserved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", nargs="*", default=["6:5", "5:6", "4:8", "3:10", "2:15"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    runtime = cfg.get("runtime", {})
    max_alloc = float(runtime.get("target_own_gpu_memory_gb", 27.0))
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    results = []
    selected = None
    for item in args.candidates:
        b, a = [int(x) for x in item.split(":")]
        try:
            result = _one_probe(cfg, b, a, device)
            if result.get("pass") and float(result.get("peak_allocated_gb", 999.0)) <= max_alloc:
                result["selected"] = True
                selected = result
                results.append(result)
                break
            result["selected"] = False
            results.append(result)
        except RuntimeError as exc:
            torch.cuda.empty_cache()
            results.append({
                "batch_size": b,
                "gradient_accumulation_steps": a,
                "effective_batch": b * a,
                "pass": False,
                "selected": False,
                "error": str(exc).splitlines()[0],
            })
    payload = {
        "pass": selected is not None,
        "selected_batch_size": selected["batch_size"] if selected else None,
        "selected_gradient_accumulation_steps": selected["gradient_accumulation_steps"] if selected else None,
        "target_own_gpu_memory_gb": max_alloc,
        "candidates": args.candidates,
        "results": results,
        "probe_type": "real_cuda_forward_backward",
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "VISTA_MEMORY_PASS.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
