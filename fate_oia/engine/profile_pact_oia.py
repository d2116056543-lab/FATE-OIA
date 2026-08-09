from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_pact_oia_probe import build_model, load_config, load_source, make_dataset, make_loader
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.utils.pact_artifacts import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--batches", type=int, nargs="+", default=[6, 5, 4])
    args = parser.parse_args(); cfg = load_config(args.config); device = torch.device(args.device)
    dataset = make_dataset(cfg, "train"); rows = []
    for batch_size in args.batches:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        model = build_model(cfg, device); load_source(model, args.source_checkpoint); model.train()
        loader = make_loader(Subset(dataset, list(range(batch_size * 4))), batch_size, False, 8, cfg)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-5)
        timings = []
        try:
            for index, batch in enumerate(loader):
                torch.cuda.synchronize(device); started = time.perf_counter()
                image, target = batch["image"].to(device, non_blocking=True), batch["action"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    output = model(image, semantic_share_license=0.5, action_scale=1.0, reason_budget=0.55)
                    loss = asymmetric_loss_with_logits(output["action_logits_final"], target)
                loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device); timings.append(time.perf_counter() - started)
            peak = torch.cuda.max_memory_reserved(device) / 1024 ** 3
            rows.append({"batch_size": batch_size, "status": "ok", "peak_reserved_gb": peak,
                         "mean_step_seconds": sum(timings[1:] or timings) / len(timings[1:] or timings),
                         "samples_per_second": batch_size / (sum(timings[1:] or timings) / len(timings[1:] or timings))})
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size": batch_size, "status": "oom"})
        del model, optimizer, loader
    valid = [row for row in rows if row["status"] == "ok" and row["peak_reserved_gb"] <= cfg["runtime"]["max_reserved_memory_gb"]]
    selected = max(valid, key=lambda row: row["samples_per_second"]) if valid else None
    write_json(Path(args.output), {"real_dino": True, "precision": "bf16", "num_workers": 8,
                                  "candidates": rows, "selected": selected})
    if selected is None:
        raise SystemExit("No safe runtime profile candidate")


if __name__ == "__main__":
    main()
