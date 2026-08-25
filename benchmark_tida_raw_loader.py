from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset, tida_video_collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument(
        "--manifest",
        default="artifacts/tida_10k_v8/tida_10k_primary_manifest.jsonl",
    )
    parser.add_argument(
        "--object-track-store",
        default=r"F:\FATE_Drive_runs\tida_object_tracks_1000_calib324_test885.pt",
    )
    parser.add_argument(
        "--frame-store-root",
        default=r"F:\FATE_Drive_runs\tida_raw_frames_1000_calib324_test885",
    )
    args = parser.parse_args()
    dataset = BDDOIAVideoDataset(
        args.manifest,
        "train_core",
        training=True,
        max_samples=1000,
        object_track_store_path=args.object_track_store,
        frame_store_root=args.frame_store_root,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": 6,
        "shuffle": True,
        "num_workers": args.workers,
        "pin_memory": True,
        "collate_fn": tida_video_collate,
        "generator": torch.Generator().manual_seed(20260825),
    }
    if args.workers:
        kwargs.update(prefetch_factor=4, persistent_workers=True)
    loader = DataLoader(**kwargs)
    timings = []
    started = time.perf_counter()
    previous = started
    for batch_index, _batch in enumerate(loader):
        now = time.perf_counter()
        if batch_index >= 2:
            timings.append(now - previous)
        previous = now
        if batch_index + 1 >= args.batches:
            break
    print(json.dumps({
        "workers": args.workers,
        "startup_and_batches_seconds": time.perf_counter() - started,
        "steady_batch_median_seconds": statistics.median(timings),
        "steady_batch_p90_seconds": sorted(timings)[max(0, int(len(timings) * 0.9) - 1)],
    }), flush=True)


if __name__ == "__main__":
    main()
