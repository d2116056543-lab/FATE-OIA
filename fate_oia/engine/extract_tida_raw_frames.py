from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from fate_oia.datasets.bdd_oia_video import quadratic_multirate_timestamps, timestamps_to_indices
from fate_oia.datasets.tida_clip_manifest import load_manifest


def decode_fixed_history(clip_path: Path, indices: torch.Tensor) -> list[Image.Image]:
    import imageio_ffmpeg

    generator = imageio_ffmpeg.read_frames(str(clip_path), pix_fmt="rgb24")
    metadata = next(generator)
    width, height = metadata["size"]
    wanted = {int(value): position for position, value in enumerate(indices.tolist())}
    frames: list[Image.Image | None] = [None] * len(indices)
    try:
        for frame_index, raw in enumerate(generator):
            position = wanted.get(frame_index)
            if position is not None:
                array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                frames[position] = Image.fromarray(array.copy(), "RGB")
            if frame_index >= int(indices.max()):
                break
    finally:
        generator.close()
    if any(frame is None for frame in frames):
        raise RuntimeError(f"ffmpeg did not decode every requested frame: {clip_path}")
    return [frame for frame in frames if frame is not None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--track-store", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    track_payload = torch.load(args.track_store, map_location="cpu", weights_only=True)
    selected = {str(name).lower() for name in track_payload["file_names"]}
    records = [row for row in load_manifest(args.manifest) if row.file_name.lower() in selected]
    if len(records) != len(selected):
        raise RuntimeError("raw-frame extraction records do not match track store")
    if args.max_samples is not None:
        records = records[: args.max_samples]
    output_root = Path(args.output_root)
    started = time.perf_counter()
    def process(record):
        case_dir = output_root / record.partition / Path(record.file_name).stem
        marker = case_dir / "COMPLETE.json"
        if marker.exists() and all((case_dir / f"{position:02d}.jpg").is_file() for position in range(14)):
            return record.file_name, True
        indices = timestamps_to_indices(
            quadratic_multirate_timestamps(), record.fps, record.target_frame_index
        )[:-1]
        frames = decode_fixed_history(record.clip_path, indices)
        case_dir.mkdir(parents=True, exist_ok=True)
        for position, frame in enumerate(frames):
            temporary = case_dir / f"{position:02d}.jpg.tmp"
            frame.save(temporary, format="JPEG", quality=args.jpeg_quality)
            temporary.replace(case_dir / f"{position:02d}.jpg")
        marker.write_text(json.dumps({
            "file_name": record.file_name, "clip_path": str(record.clip_path),
            "frame_indices": indices.tolist(), "decoder": "imageio_ffmpeg_subprocess",
        }), encoding="utf-8")
        return record.file_name, False

    complete = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(process, records)
        for record_index, (_file_name, _cached) in enumerate(results, start=1):
            complete += 1
            if complete == 1 or complete % 50 == 0 or record_index == len(records):
                elapsed = time.perf_counter() - started
                print(json.dumps({
                    "complete": complete, "total": len(records),
                    "samples_per_second": complete / max(elapsed, 1e-6),
                }), flush=True)
    print(json.dumps({"pass": complete == len(records), "complete": complete,
                      "total": len(records), "output_root": str(output_root)}), flush=True)


if __name__ == "__main__":
    main()
