from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from fate_oia.datasets.bdd_oia_video import (
    decode_selected_frames,
    quadratic_multirate_timestamps,
    timestamps_to_indices,
)


def sequential_decode(path: str, indices: torch.Tensor):
    import cv2
    from PIL import Image

    requested = [int(value) for value in indices.tolist()]
    capture = cv2.VideoCapture(path)
    frames = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, requested[0])
        position = requested[0]
        wanted = set(requested)
        decoded = {}
        while position <= requested[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if position in wanted:
                decoded[position] = Image.fromarray(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                )
            position += 1
        valid = torch.tensor([value in decoded for value in requested])
        previous = None
        for value in requested:
            if value in decoded:
                previous = decoded[value]
            if previous is None:
                previous = Image.new("RGB", (1280, 720))
            frames.append(previous.copy())
        return frames, valid
    finally:
        capture.release()


def decord_decode(path: str, indices: torch.Tensor):
    from decord import VideoReader, cpu
    from PIL import Image

    reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    arrays = reader.get_batch(indices.tolist()).asnumpy()
    return [Image.fromarray(array) for array in arrays], torch.ones(
        len(indices), dtype=torch.bool
    )


def frame_mse(left, right) -> float:
    values = []
    for first, second in zip(left, right):
        a = np.asarray(first, dtype=np.float32) / 255.0
        b = np.asarray(second, dtype=np.float32) / 255.0
        values.append(float(np.square(a - b).mean()))
    return max(values, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--history-frames", type=int, default=5)
    args = parser.parse_args()

    rows = []
    with Path(args.manifest).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("history_available", True):
                rows.append(row)
            if len(rows) >= args.samples:
                break
    timings = {"sparse_seek": 0.0, "sequential": 0.0, "decord": 0.0}
    max_mse = 0.0
    for row in rows:
        timestamps = quadratic_multirate_timestamps(args.history_frames + 1)
        indices = timestamps_to_indices(
            timestamps, float(row["fps"]), int(row["target_frame_index"])
        )[:-1]
        start = time.perf_counter()
        sparse_frames, sparse_valid = decode_selected_frames(row["clip_path"], indices)
        timings["sparse_seek"] += time.perf_counter() - start
        start = time.perf_counter()
        sequential_frames, sequential_valid = sequential_decode(row["clip_path"], indices)
        timings["sequential"] += time.perf_counter() - start
        if not torch.equal(sparse_valid, sequential_valid):
            raise RuntimeError("decoder validity mismatch")
        max_mse = max(max_mse, frame_mse(sparse_frames, sequential_frames))
        start = time.perf_counter()
        decord_frames, decord_valid = decord_decode(row["clip_path"], indices)
        timings["decord"] += time.perf_counter() - start
        if not torch.equal(sparse_valid, decord_valid):
            raise RuntimeError("decord validity mismatch")
        max_mse = max(max_mse, frame_mse(sparse_frames, decord_frames))
    print(json.dumps({
        "samples": len(rows),
        "history_frames": args.history_frames,
        "seconds": timings,
        "clips_per_second": {
            key: len(rows) / max(value, 1e-9) for key, value in timings.items()
        },
        "max_frame_mse": max_mse,
    }, indent=2))


if __name__ == "__main__":
    main()
