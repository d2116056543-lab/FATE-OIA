from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from fate_oia.datasets.tida_clip_manifest import compare_last_frames


def frame_window(target_frame: int, fps: float, seconds: float) -> tuple[int, int, int]:
    end = int(target_frame)
    start = max(0, end - int(round(float(fps) * float(seconds))))
    return start, end, end - start + 1


def ffmpeg_select_filter(start: int, end: int, fps: float) -> str:
    return f"select=between(n\\,{int(start)}\\,{int(end)}),setpts=N/{float(fps):.12f}/TB"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _metadata_and_endpoint(path: Path) -> tuple[float, int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
    ok, endpoint = capture.read()
    capture.release()
    if not ok or fps <= 0 or frames < 2:
        raise RuntimeError(f"repaired clip is not decodable: {path}")
    return fps, frames, cv2.cvtColor(endpoint, cv2.COLOR_BGR2RGB)


def repair_row(row: dict[str, Any], output_root: Path, ffmpeg: str, seconds: float) -> dict[str, Any]:
    fps = float(row["match"]["fps"])
    start, end, expected = frame_window(int(row["match"]["frame"]), fps, seconds)
    output = output_root / str(row["split"]) / f"{Path(row['image_name']).stem}_repaired.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    video_filter = ffmpeg_select_filter(start, end, fps)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(row["video_path"]),
        "-vf", video_filter, "-an", "-c:v", "mpeg4", "-q:v", "2", "-pix_fmt", "yuv420p", str(output),
    ]
    subprocess.run(command, check=True)
    output_fps, frames, endpoint = _metadata_and_endpoint(output)
    reference = np.asarray(Image.open(row["last_frame_image_path"]).convert("RGB"))
    alignment = compare_last_frames(reference, endpoint)
    if abs(frames - expected) > 1 or not alignment["pass"]:
        raise RuntimeError(
            f"repaired endpoint mismatch for {row['image_name']}: frames={frames}/{expected}, alignment={alignment}"
        )
    return {
        "image_name": row["image_name"],
        "original_clip_path": row["clip_path"],
        "clip_path": str(output.resolve()),
        "fps": output_fps,
        "frames_written": frames,
        "expected_frames": expected,
        "alignment": alignment,
        "ffmpeg_command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()
    import imageio_ffmpeg

    repaired = []
    for manifest_value in args.source_manifest:
        for row in _read_rows(Path(manifest_value)):
            if int(row.get("clip", {}).get("frames_written", 0)) < 2:
                repaired.append(
                    repair_row(row, Path(args.output_root), imageio_ffmpeg.get_ffmpeg_exe(), args.seconds)
                )
    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in repaired), encoding="utf-8"
    )
    print(json.dumps({"pass": True, "repaired": len(repaired), "manifest": str(output_manifest.resolve())}))


if __name__ == "__main__":
    main()
