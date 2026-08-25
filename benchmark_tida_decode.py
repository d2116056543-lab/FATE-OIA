from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import cv2
import torch

from fate_oia.datasets.bdd_oia_video import (
    _decode_selected_frames_from_capture,
    decode_selected_frames,
    quadratic_multirate_timestamps,
    timestamps_to_indices,
)


def sequential_decode(path: str, indices: torch.Tensor):
    capture = cv2.VideoCapture(path)
    try:
        return _decode_selected_frames_from_capture(
            capture,
            indices,
            bgr_to_rgb=lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
    finally:
        capture.release()


manifest = Path("artifacts/tida_10k_v8/tida_10k_primary_manifest.jsonl")
groups: dict[str, list[dict]] = {"F": [], "G": []}
for line in manifest.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    drive = Path(row["clip_path"]).drive.rstrip(":").upper()
    if drive in groups and len(groups[drive]) < 6:
        groups[drive].append(row)

result = {}
for drive, rows in groups.items():
    old_times = []
    new_times = []
    valid_counts = []
    for row in rows:
        indices = timestamps_to_indices(
            quadratic_multirate_timestamps(), row["fps"], row["target_frame_index"]
        )[:-1]
        started = time.perf_counter()
        _, old_valid = sequential_decode(row["clip_path"], indices)
        old_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        _, new_valid = decode_selected_frames(row["clip_path"], indices)
        new_times.append(time.perf_counter() - started)
        valid_counts.append((int(old_valid.sum()), int(new_valid.sum())))
    result[drive] = {
        "samples": len(rows),
        "sequential_median_seconds": statistics.median(old_times),
        "hybrid_median_seconds": statistics.median(new_times),
        "speedup": statistics.median(old_times) / max(statistics.median(new_times), 1e-9),
        "valid_counts": valid_counts,
    }
print(json.dumps(result, ensure_ascii=False, indent=2))
