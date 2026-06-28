from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--target_min_gb", type=float, default=40.0)
    ap.add_argument("--target_max_gb", type=float, default=45.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from fate_oia.engine.train_pmcal_v2_oia import load_config, build_model, make_loader, loss_bundle

    cfg = load_config(args.config)
    ladder = cfg.get("training", {}).get("memory_probe_ladder", [[9, 4], [8, 4], [7, 5], [6, 5], [5, 6], [4, 8], [3, 11], [2, 16]])
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    attempts = []
    selected = None
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for batch_size, accum in ladder:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            model = build_model(cfg, device)
            loader = make_loader(cfg, "train", int(batch_size), int(batch_size), False, 0)
            batch = next(iter(loader))
            images = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out = model(images, split="train", action_labels=action, reason_labels=reason, file_names=batch["file_name"])
            loss, stats = loss_bundle(out, action, reason, cfg)
            loss.backward()
            allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3) if device.type == "cuda" else 0.0
            ok = reserved <= float(args.target_max_gb)
            attempts.append({"batch_size": int(batch_size), "grad_accum": int(accum), "ok": bool(ok), "peak_allocated_gb": allocated, "reserved_gb": reserved, "loss": float(loss.detach().cpu())})
            del model, out, loss, batch, images, action, reason
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if ok and selected is None:
                selected = attempts[-1]
                if allocated >= float(args.target_min_gb) * 0.75:
                    break
        except RuntimeError as exc:
            msg = str(exc)
            attempts.append({"batch_size": int(batch_size), "grad_accum": int(accum), "ok": False, "oom": "out of memory" in msg.lower(), "error": msg[:300]})
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
    if selected is None:
        selected = next((a for a in reversed(attempts) if a.get("ok")), attempts[-1] if attempts else {"batch_size": 1, "grad_accum": 30, "ok": False})
    result = {
        "selected_batch_size": int(selected.get("batch_size", 1)),
        "selected_gradient_accumulation_steps": int(selected.get("grad_accum", 30)),
        "peak_allocated_gb": float(selected.get("peak_allocated_gb", 0.0)),
        "reserved_gb": float(selected.get("reserved_gb", 0.0)),
        "oom_attempts": [a for a in attempts if a.get("oom")],
        "fallback_attempts": attempts,
        "forward_backward_proof": bool(selected.get("ok", False)),
        "warning": "" if selected.get("ok", False) else "no safe candidate selected",
    }
    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
