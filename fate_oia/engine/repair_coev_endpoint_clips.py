from __future__ import annotations

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image


def _rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _stem(file_name: str) -> str:
    value = Path(file_name).stem
    return value[:-2] if value.endswith(("_1", "_3")) else value


def _download_raw(record: dict, output: Path) -> None:
    temporary = output.with_suffix(output.suffix + ".part")
    with temporary.open("wb") as target:
        for segment in record["segments"]:
            expected = int(segment["end"]) - int(segment["start"]) + 1
            request = urllib.request.Request(
                record["part_urls"][str(segment["part"])],
                headers={"Range": f"bytes={segment['start']}-{segment['end']}", "User-Agent": "coev-endpoint-repair/1"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                written = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    written += len(chunk)
            if written != expected:
                raise RuntimeError(f"range length mismatch {written} != {expected}")
        target.flush()
        os.fsync(target.fileno())
    if temporary.stat().st_size != int(record["uncompressed_size"]):
        raise RuntimeError("downloaded raw video size differs from ZIP metadata")
    temporary.replace(output)


def _write_clip(frames: list, fps: float, output: Path) -> Path:
    temporary = output.with_suffix(".repair.mp4")
    temporary.unlink(missing_ok=True)
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(fps)).limit_denominator(100000))
        stream.width = frames[0].width
        stream.height = frames[0].height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12", "preset": "fast"}
        for frame in frames:
            clean = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
            for packet in stream.encode(clean):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return temporary


def _frame_mse(frame, target_path: Path) -> float:
    target = Image.open(target_path).convert("RGB")
    decoded = frame.to_image().convert("RGB")
    left = np.asarray(decoded.resize(target.size), dtype=np.float32) / 255.0
    right = np.asarray(target, dtype=np.float32) / 255.0
    return float(np.square(left - right).mean())


def _mse(clip: Path, target_path: Path, target_index: int) -> float:
    endpoint = None
    with av.open(str(clip)) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index == target_index:
                endpoint = frame.to_image().convert("RGB")
                break
    if endpoint is None:
        raise RuntimeError("repaired clip does not contain target frame")
    target = Image.open(target_path).convert("RGB")
    left = np.asarray(endpoint.resize(target.size), dtype=np.float32) / 255.0
    right = np.asarray(target, dtype=np.float32) / 255.0
    return float(np.square(left - right).mean())


def repair_stem(stem: str, jobs: list[dict], range_record: dict, temporary_root: Path, max_mse: float) -> list[dict]:
    raw = temporary_root / f"{stem}.mov"
    _download_raw(range_record, raw)
    required = {index for job in jobs for index in range(job["start_frame"], job["end_frame"] + 1)}
    decoded = {}
    fps = None
    try:
        with av.open(str(raw)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            for index, frame in enumerate(container.decode(stream)):
                if index in required:
                    decoded[index] = frame
        results = []
        for job in jobs:
            sequence = [decoded[index] for index in range(job["start_frame"], job["end_frame"] + 1)]
            if len(sequence) != job["frames_written"]:
                raise RuntimeError(f"raw decode missing requested frames for {job['file_name']}")
            destination = Path(job["clip_path"]);target_path = Path(job["target_image_path"])
            raw_mse = _frame_mse(sequence[-1], target_path)
            if raw_mse > max_mse:
                raise RuntimeError(f"downloaded raw endpoint MSE {raw_mse:.6f} exceeds {max_mse}")
            temporary = _write_clip(sequence, fps, destination)
            endpoint_mse = _mse(temporary, target_path, job["frames_written"] - 1)
            if endpoint_mse > max_mse:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"re-encoded endpoint MSE {endpoint_mse:.6f} exceeds {max_mse}")
            temporary.replace(destination)
            results.append({"file_name": job["file_name"], "clip_path": str(destination),
                            "raw_endpoint_mse": raw_mse, "endpoint_mse": endpoint_mse})
        return results
    finally:
        raw.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-audit", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--materialize-status", required=True)
    parser.add_argument("--range-manifest", required=True)
    parser.add_argument("--temporary-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-mse", type=float, default=0.01)
    parser.add_argument("--max-stems", type=int)
    args = parser.parse_args()
    endpoint = json.loads(Path(args.endpoint_audit).read_text(encoding="utf-8"))
    bad = {row["file_name"] for row in endpoint["mismatches"]}
    manifest = {row["file_name"]: row for row in _rows(Path(args.manifest))}
    statuses = {row["image_name"]: row for row in _rows(Path(args.materialize_status)) if row.get("ok")}
    ranges = {row["stem"]: row for row in _rows(Path(args.range_manifest))}
    grouped = defaultdict(list)
    for file_name in sorted(bad & statuses.keys() & manifest.keys()):
        status, identity = statuses[file_name], manifest[file_name]
        stem = _stem(file_name)
        if stem not in ranges:
            continue
        grouped[stem].append({
            "file_name": file_name,
            "clip_path": identity["clip_path"],
            "target_image_path": identity["target_image_path"],
            "start_frame": int(status["clip"]["start_frame"]),
            "end_frame": int(status["clip"]["end_frame"]),
            "frames_written": int(status["clip"]["frames_written"]),
        })
    stems = sorted(grouped)
    if args.max_stems is not None:
        stems = stems[: args.max_stems]
    output = Path(args.output)
    prior = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {"repaired": [], "failures": []}
    completed = {row["stem"] for row in prior["repaired"] + prior["failures"]}
    temporary_root = Path(args.temporary_root)
    temporary_root.mkdir(parents=True, exist_ok=True)
    for index, stem in enumerate(stems, 1):
        if stem in completed:
            continue
        try:
            rows = repair_stem(stem, grouped[stem], ranges[stem], temporary_root, args.max_mse)
            prior["repaired"].append({"stem": stem, "clips": rows})
        except Exception as error:
            prior["failures"].append({"stem": stem, "error": repr(error)})
        prior.update({"schema": "coev_endpoint_repair_v1", "candidate_stems": len(grouped), "requested_stems": len(stems)})
        temporary = output.with_suffix(output.suffix + ".tmp")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(output)
        print(json.dumps({"done": index, "total": len(stems), "stem": stem}), flush=True)
    if prior["failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
