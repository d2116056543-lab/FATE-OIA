from __future__ import annotations

from pathlib import Path


class PSIFrameResolver:
    """Resolve PSI frame paths without ever adding the target frame to inputs."""

    def __init__(self, frames_root: str | Path) -> None:
        self.frames_root = Path(frames_root)

    def resolve(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        direct = self.frames_root / path
        if direct.exists():
            return direct
        nested = self.frames_root / "frames" / path
        if nested.exists():
            return nested
        return direct

    def resolve_frame_id(self, video_id: str, frame_id: int | str) -> Path:
        fid = int(frame_id)
        root = self.frames_root
        if (root / "frames").exists():
            root = root / "frames"
        return root / str(video_id) / f"{fid:03d}.jpg"

    def resolve_sequence(self, frame_values: list[str | Path | int], expected_count: int = 15, video_id: str | None = None) -> list[Path]:
        paths: list[Path] = []
        for value in frame_values:
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                if video_id is None:
                    raise ValueError("numeric PSI frame ids require video_id")
                paths.append(self.resolve_frame_id(video_id, value))
            else:
                paths.append(self.resolve(value))
        if len(paths) != expected_count:
            raise ValueError(f"PSI formal input requires {expected_count} observed frames, got {len(paths)}")
        return paths


def assert_target_not_in_inputs(input_paths: list[Path], target_path: str | Path | None) -> None:
    if target_path is None:
        return
    target = Path(target_path)
    target_name = target.as_posix().lower()
    inputs = {p.as_posix().lower() for p in input_paths}
    if target_name in inputs:
        raise ValueError("target_frame image leaked into formal input_frames")
