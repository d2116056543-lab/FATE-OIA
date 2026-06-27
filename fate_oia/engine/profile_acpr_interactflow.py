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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_interactflow_config(args.config)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    ds = PSIDAMO11902Dataset(
        cfg["paths"]["psi_package_root"],
        "train",
        frames_root=cfg["paths"].get("psi2_root_reference_only"),
        image_size=(int(cfg["data"]["image_height"]), int(cfg["data"]["image_width"])),
        action_dim=int(cfg["data"]["action_dim"]),
        max_samples=max(args.batch_size * args.measured_batches, args.batch_size),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=psi_interactflow_collate)
    model = ACPRInteractFlowPPModel(
        pretrained_weights=cfg["paths"]["dino_weights"],
        predicate_config="configs/acpr_interactflow_predicates.yaml",
        grammar_path=cfg["model"]["interaction_flow"]["grammar_yaml"],
        action_dim=int(cfg["data"]["action_dim"]),
        use_mock_dino=False,
    ).to(device).eval()
    start = time.time()
    load_time = 0.0
    forward_time = 0.0
    count = 0
    next_start = time.time()
    with torch.no_grad():
        for batch in loader:
            load_time += time.time() - next_start
            frames = batch.input_frames.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            fwd_start = time.time()
            _ = model(frames)
            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time += time.time() - fwd_start
            count += 1
            if count >= args.measured_batches:
                break
            next_start = time.time()
    peak = torch.cuda.max_memory_reserved() / (1024 ** 3) if device.type == "cuda" else 0.0
    report = {
        "measured_batches": args.measured_batches,
        "profile_kind": "real_dataset_forward_profile",
        "completed_batches": count,
        "batch_size": args.batch_size,
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
