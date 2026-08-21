from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from fate_oia.datasets.tida_clip_manifest import (
    TIDAClipRecord,
    _phash64,
    file_sha256,
    normalize_source_id,
    partition_train_records,
    validate_records,
    write_manifest,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("records", "samples", "items"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"unsupported source manifest schema: {path}")


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _video_metadata(path: Path) -> tuple[float, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or fps <= 0 or frames <= 0:
        capture.release()
        raise ValueError(f"undecodable clip: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
    ok, endpoint = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode final frame: {path}")
    endpoint_rgb = cv2.cvtColor(endpoint, cv2.COLOR_BGR2RGB)
    return fps, frames / fps, frames, _phash64(endpoint_rgb)


def build_records(source_manifests: list[Path]) -> list[TIDAClipRecord]:
    records: list[TIDAClipRecord] = []
    for source_manifest in source_manifests:
        rows = _load_rows(source_manifest)
        for row_index, row in enumerate(rows):
            if "split" not in row or "image_name" not in row:
                raise ValueError(f"source row missing split/image_name: {source_manifest}:{row_index}")
            official_split = str(row["split"]).lower()
            target_value = row.get("last_frame_image_path") or row.get("image_path")
            clip_value = row.get("clip_path") or row.get("video_path")
            if not target_value or not clip_value:
                raise ValueError(f"source row missing target/clip path: {source_manifest}:{row_index}")
            target_path = _resolve_path(str(target_value), source_manifest.parent)
            clip_path = _resolve_path(str(clip_value), source_manifest.parent)
            action = row.get("action_4") or row.get("action")
            reason = row.get("reason_21") or row.get("reason")
            if action is None or reason is None:
                raise ValueError(f"source row missing labels: {source_manifest}:{row_index}")
            fps, duration, num_frames, endpoint_phash = _video_metadata(clip_path)
            source_value = row.get("source_video_id") or row.get("source_video") or row.get("video_path") or clip_path.stem
            records.append(
                TIDAClipRecord(
                    official_split=official_split,
                    partition="test" if official_split == "test" else "unassigned",
                    file_name=str(row["image_name"]),
                    target_image_path=target_path,
                    clip_path=clip_path,
                    source_video_id=normalize_source_id(str(source_value)),
                    duration_seconds=duration,
                    fps=fps,
                    num_frames=num_frames,
                    target_timestamp_seconds=duration,
                    target_frame_index=num_frames - 1,
                    action=tuple(float(value) for value in action),
                    reason=tuple(float(value) for value in reason),
                    clip_sha256=file_sha256(clip_path),
                    endpoint_phash=endpoint_phash,
                    source_batch=source_manifest.parent.name,
                    source_manifest_path=str(source_manifest.resolve()),
                    source_row_index=row_index,
                )
            )
    train = partition_train_records([record for record in records if record.official_split == "train"])
    test = [record for record in records if record.official_split == "test"]
    result = train + sorted(test, key=lambda record: record.file_name)
    validation = validate_records(result, require_files=True)
    if not validation["pass"]:
        raise RuntimeError(f"TIDA manifest validation failed: {validation['errors']}")
    counts = {partition: sum(record.partition == partition for record in result) for partition in ("train_core", "train_calib", "train_audit", "test")}
    if counts != {"train_core": 2291, "train_calib": 312, "train_audit": 512, "test": 885}:
        raise RuntimeError(f"formal partition counts differ: {counts}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = build_records([Path(value) for value in args.source_manifest])
    write_manifest(args.output, records)
    print(json.dumps({"event": "tida_manifest", "count": len(records), "output": str(Path(args.output).resolve())}), flush=True)


if __name__ == "__main__":
    main()
