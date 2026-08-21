from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fate_oia.datasets.tida_clip_manifest import compare_last_frames, load_manifest, validate_records
from fate_oia.datasets.bdd_oia_video import decode_selected_frames, quadratic_multirate_timestamps, timestamps_to_indices
from fate_oia.utils.tida_artifacts import append_jsonl, atomic_write_json, file_sha256


def decode_last_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode endpoint: {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _audit_one(payload: tuple[int, object]) -> tuple[dict, int, int, bool]:
    index, row = payload
    reference = np.asarray(Image.open(row.target_image_path).convert("RGB"))
    indices = timestamps_to_indices(quadratic_multirate_timestamps(), row.fps, row.target_frame_index)
    selected_frames, selected_valid = decode_selected_frames(row.clip_path, indices)
    decoded_arrays = [np.asarray(frame.resize((64, 36)), dtype=np.float32) for frame in selected_frames]
    duplicate_pairs = sum(
        float(np.mean((left - right) ** 2)) < 1e-6
        for left, right in zip(decoded_arrays, decoded_arrays[1:])
    )
    score = compare_last_frames(reference, np.asarray(selected_frames[-1]))
    record = {
        "index": index, "file_name": row.file_name, "partition": row.partition,
        "selected_valid_count": int(selected_valid.sum()), "selected_frame_count": len(selected_valid),
        "adjacent_exact_duplicate_pairs": duplicate_pairs, **score,
    }
    return record, int(selected_valid.sum()), len(selected_valid), duplicate_pairs == len(decoded_arrays) - 1


def audit_manifest(manifest_path: Path, output_dir: Path, *, workers: int = 6) -> dict:
    rows = load_manifest(manifest_path)
    validation = validate_records(rows, require_files=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "last_frame_audit.jsonl"
    if sample_path.exists():
        sample_path.unlink()
    scores = []
    rejected = []
    selected_valid_count = 0
    selected_total_count = 0
    fully_repeated_clips = []
    if workers < 1:
        raise ValueError("workers must be positive")
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tida-data-audit") as executor:
        for record, valid_count, total_count, fully_repeated in executor.map(_audit_one, enumerate(rows)):
            selected_valid_count += valid_count
            selected_total_count += total_count
            if fully_repeated:
                fully_repeated_clips.append(record["file_name"])
            append_jsonl(sample_path, record)
            scores.append(record)
            if not record["pass"]:
                rejected.append(record["file_name"])
    median_ssim = float(np.median([row["ssim"] for row in scores])) if scores else 0.0
    def percentiles(key: str) -> dict[str, float]:
        values = [float(row[key]) for row in scores]
        return {name: float(np.percentile(values, value)) for name, value in (("p10", 10), ("p50", 50), ("p90", 90))} if values else {}
    counts = {partition: sum(row.partition == partition for row in rows) for partition in ("train_core", "train_calib", "train_audit", "test")}
    official_counts = {split: sum(row.official_split == split for row in rows) for split in ("train", "test")}
    split_migration = {
        "official_train_to_internal_test": sum(row.official_split == "train" and row.partition == "test" for row in rows),
        "official_test_to_internal_train": sum(row.official_split == "test" and row.partition != "test" for row in rows),
        "official_train_retained_internal_train": sum(row.official_split == "train" and row.partition != "test" for row in rows),
        "official_test_retained_internal_test": sum(row.official_split == "test" and row.partition == "test" for row in rows),
    }
    duration_rate = sum(row.duration_seconds >= 4.8 for row in rows) / max(len(rows), 1)
    mapping_unique_rate = len({(row.official_split, row.file_name) for row in rows}) / max(len(rows), 1)
    target_exists_rate = sum(row.target_image_path.is_file() for row in rows) / max(len(rows), 1)
    clip_exists_rate = sum(row.clip_path.is_file() for row in rows) / max(len(rows), 1)
    selected_decode_success_rate = selected_valid_count / max(selected_total_count, 1)
    passed = (
        validation["pass"]
        and len(rows) == 4000
        and counts == {"train_core": 2291, "train_calib": 312, "train_audit": 512, "test": 885}
        and not rejected
        and median_ssim >= 0.995
        and duration_rate >= 0.995
        and mapping_unique_rate == 1.0
        and target_exists_rate == 1.0
        and clip_exists_rate == 1.0
        and selected_decode_success_rate >= 0.995
        and not fully_repeated_clips
    )
    summary = {
        "pass": passed,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "count": len(rows),
        "partition_counts": counts,
        "official_split_counts": official_counts,
        "split_strategy": "source_grouped_internal_3115_885",
        "split_migration": split_migration,
        "publication_eligible": False,
        "audit_workers": workers,
        "median_ssim": median_ssim,
        "mapping_unique_rate": mapping_unique_rate,
        "target_image_exists_rate": target_exists_rate,
        "clip_decode_success_rate": clip_exists_rate,
        "selected_frame_decode_success_rate": selected_decode_success_rate,
        "fully_repeated_clips": fully_repeated_clips,
        "duration_at_least_4p8_rate": duration_rate,
        "exact_pixel_rate": sum(float(row["normalized_mae"]) == 0.0 for row in scores) / max(len(scores), 1),
        "ssim_percentiles": percentiles("ssim"), "psnr_percentiles": percentiles("psnr"),
        "normalized_mae_percentiles": percentiles("normalized_mae"),
        "individual_failures": rejected,
        "validation_errors": validation["errors"],
        "sample_artifact": str(sample_path.resolve()),
    }
    atomic_write_json(output_dir / "TIDA_DATA_AUDIT.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    summary = audit_manifest(Path(args.clip_manifest), Path(args.output_dir), workers=args.workers)
    print(json.dumps(summary), flush=True)
    if not summary["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
