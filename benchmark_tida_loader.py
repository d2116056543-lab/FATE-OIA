from __future__ import annotations

import json
import statistics
import time

import cv2
import torch
from torch.utils.data import DataLoader

from fate_oia.datasets.bdd_oia_video import (
    BDDOIAVideoDataset,
    _decode_selected_frames_from_capture,
    decode_selected_frames,
)


def sequential_decode(path, indices):
    capture = cv2.VideoCapture(str(path))
    try:
        return _decode_selected_frames_from_capture(
            capture,
            indices,
            bgr_to_rgb=lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
    finally:
        capture.release()


def benchmark(decoder_name: str, workers: int) -> dict:
    decoder = sequential_decode if decoder_name == "sequential" else decode_selected_frames
    dataset = BDDOIAVideoDataset(
        "artifacts/tida_10k_v8/tida_10k_primary_manifest.jsonl",
        "train_core",
        training=True,
        decoder=decoder,
        max_samples=120,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": 6,
        "shuffle": True,
        "num_workers": workers,
        "pin_memory": True,
        "generator": torch.Generator().manual_seed(20260825),
    }
    if workers:
        kwargs.update(prefetch_factor=2, persistent_workers=True)
    loader = DataLoader(**kwargs)
    timings = []
    previous = time.perf_counter()
    for batch_index, _batch in enumerate(loader):
        now = time.perf_counter()
        if batch_index >= 2:
            timings.append(now - previous)
        previous = now
        if batch_index >= 9:
            break
    return {
        "decoder": decoder_name,
        "workers": workers,
        "measured_batches": len(timings),
        "batch_median_seconds": statistics.median(timings),
        "batch_p90_seconds": sorted(timings)[max(0, int(len(timings) * 0.9) - 1)],
    }


if __name__ == "__main__":
    results = []
    for decoder_name in ("sequential", "hybrid"):
        for workers in (2, 4, 6):
            results.append(benchmark(decoder_name, workers))
            print(json.dumps(results[-1]), flush=True)
