from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fate_oia.acpr_interactflow.artifacts import write_json
from fate_oia.acpr_interactflow.config import load_interactflow_config
from fate_oia.acpr_interactflow.model import ACPRInteractFlowPPModel
from fate_oia.acpr_interactflow.psi_damo_dataset import PSIDAMO11902Dataset, psi_interactflow_collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/acpr_interactflow_pp_v1_psi_damo_11902.yaml")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--measured_batches", type=int, default=100)
    parser.add_argument("--warmup_batches", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_interactflow_config(args.config)
    warmup_batches = int(args.warmup_batches if args.warmup_batches is not None else cfg.get("profile", {}).get("warmup_batches", 0))
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ds = PSIDAMO11902Dataset(
        cfg["paths"]["psi_package_root"],
        "train",
        frames_root=cfg["paths"].get("psi2_root_reference_only"),
        image_size=(int(cfg["data"]["image_height"]), int(cfg["data"]["image_width"])),
        action_dim=int(cfg["data"]["action_dim"]),
        max_samples=max(args.batch_size * (args.measured_batches + warmup_batches + 1), args.batch_size),
    )
    data_cfg = cfg["data"]
    workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "collate_fn": psi_interactflow_collate,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", False))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    loader = DataLoader(ds, **loader_kwargs)
    pred_cfg = cfg["model"].get("predicates", {})
    model = ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        exp29_names_path=cfg["paths"].get("psi_label_embedding_json"),
        oia_acpr_checkpoint=cfg["paths"].get("oia_acpr_checkpoint"),
        text_encoder_model=cfg["paths"].get("text_encoder_model"),
        require_oia_transfer_source=bool(pred_cfg.get("require_oia_transfer_source", False)),
        require_transformer_text=bool(pred_cfg.get("require_transformer_text", False)),
        action_dim=int(cfg["data"]["action_dim"]),
        dino_chunk_size=int(cfg["model"]["visual_encoder"].get("dino_chunk_size", 2)),
        use_mock_dino=False,
    ).to(device).eval()
    start = time.time()
    load_time = 0.0
    forward_time = 0.0
    count = 0
    warmup_count = 0
    next_start = time.time()
    with torch.no_grad():
        for batch in loader:
            current_load_gap = time.time() - next_start
            frames = batch.input_frames.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            fwd_start = time.time()
            _ = model(frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            current_forward = time.time() - fwd_start
            if warmup_count < warmup_batches:
                warmup_count += 1
                next_start = time.time()
                continue
            load_time += current_load_gap
            forward_time += current_forward
            count += 1
            if count >= args.measured_batches:
                break
            next_start = time.time()
    peak = torch.cuda.max_memory_reserved() / (1024 ** 3) if device.type == "cuda" else 0.0
    report = {
        "measured_batches": args.measured_batches,
        "warmup_batches": warmup_batches,
        "completed_warmup_batches": warmup_count,
        "profile_kind": "real_dataset_forward_profile",
        "completed_batches": count,
        "batch_size": args.batch_size,
        "num_workers": workers,
        "pin_memory": bool(data_cfg.get("pin_memory", False)),
        "persistent_workers": bool(data_cfg.get("persistent_workers", False)) if workers > 0 else False,
        "prefetch_factor": int(data_cfg.get("prefetch_factor", 2)) if workers > 0 else None,
        "device": str(device),
        "use_mock_dino": False,
        "data_time_sec": load_time,
        "forward_time_sec": forward_time,
        "data_time_fraction": load_time / max(load_time + forward_time, 1e-9),
        "samples_per_second": (count * args.batch_size) / max(load_time + forward_time, 1e-9),
        "peak_reserved_gib": peak,
        "elapsed_sec": time.time() - start,
        "dummy_allocation_forbidden": True,
    }
    write_json(Path(args.output_dir) / "throughput_memory_profile.json", report)


if __name__ == "__main__":
    main()
