from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PARTITIONS = ("train_core", "train_calib", "train_audit", "test")


def normalize_source_id(value: str) -> str:
    stem = Path(value).stem.lower().strip()
    for suffix in ("_prev5s", "-prev5s"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


@dataclass(frozen=True)
class TIDAClipRecord:
    official_split: str
    partition: str
    file_name: str
    target_image_path: Path
    clip_path: Path
    source_video_id: str
    duration_seconds: float
    fps: float
    num_frames: int
    target_timestamp_seconds: float
    target_frame_index: int
    action: tuple[float, ...]
    reason: tuple[float, ...]
    clip_sha256: str = ""
    endpoint_phash: int = 0
    source_batch: str = ""
    source_manifest_path: str = ""
    source_row_index: int = -1

    def __post_init__(self) -> None:
        if self.official_split not in ("train", "test"):
            raise ValueError(f"invalid official_split: {self.official_split}")
        if self.partition not in PARTITIONS and self.partition != "unassigned":
            raise ValueError(f"invalid partition: {self.partition}")
        if (self.official_split == "test") != (self.partition == "test") and self.partition != "unassigned":
            raise ValueError("official_split and partition disagree")
        if len(self.action) != 4 or len(self.reason) != 21:
            raise ValueError("TIDA requires action_4 and reason_21")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TIDAClipRecord":
        if "split" in value:
            raise ValueError("legacy split field is forbidden; use official_split and partition")
        required = {
            "official_split", "partition", "file_name", "target_image_path", "clip_path",
            "source_video_id", "duration_seconds", "fps", "num_frames",
            "target_timestamp_seconds", "target_frame_index", "action", "reason",
        }
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"manifest row missing fields: {missing}")
        payload = dict(value)
        payload["target_image_path"] = Path(payload["target_image_path"])
        payload["clip_path"] = Path(payload["clip_path"])
        payload["action"] = tuple(float(x) for x in payload["action"])
        payload["reason"] = tuple(float(x) for x in payload["reason"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_image_path"] = str(self.target_image_path)
        value["clip_path"] = str(self.clip_path)
        value["action"] = list(self.action)
        value["reason"] = list(self.reason)
        return value


def _group_order(groups: dict[str, list[TIDAClipRecord]], seed: int) -> list[tuple[str, list[TIDAClipRecord]]]:
    return sorted(groups.items(), key=lambda item: (sha256(f"{seed}:{item[0]}".encode()).hexdigest(), item[0]))


def _exact_group_subset(ordered: Sequence[tuple[str, list[TIDAClipRecord]]], target: int) -> tuple[int, ...]:
    # The first path discovered under sorted traversal is the lexicographically smallest index tuple.
    paths: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, rows) in enumerate(ordered):
        size = len(rows)
        for total, path in sorted(list(paths.items()), reverse=True):
            new_total = total + size
            if new_total <= target and new_total not in paths:
                paths[new_total] = path + (index,)
    if target not in paths:
        raise ValueError(f"cannot form exact group-safe partition of {target} records")
    return paths[target]


def partition_train_records(
    records: Sequence[TIDAClipRecord], *, seed: int = 20260821, calib_count: int = 312, audit_count: int = 512
) -> list[TIDAClipRecord]:
    groups: dict[str, list[TIDAClipRecord]] = {}
    for record in records:
        if record.official_split != "train":
            raise ValueError("partition_train_records accepts official train records only")
        groups.setdefault(normalize_source_id(record.source_video_id), []).append(record)
    ordered = _group_order(groups, seed)
    calib_indices = set(_exact_group_subset(ordered, calib_count))
    remaining = [entry for i, entry in enumerate(ordered) if i not in calib_indices]
    audit_remaining_indices = set(_exact_group_subset(remaining, audit_count))
    calib_ids = {ordered[i][0] for i in calib_indices}
    audit_ids = {remaining[i][0] for i in audit_remaining_indices}
    result = []
    for record in records:
        source_id = normalize_source_id(record.source_video_id)
        partition = "train_calib" if source_id in calib_ids else "train_audit" if source_id in audit_ids else "train_core"
        result.append(replace(record, partition=partition))
    return sorted(result, key=lambda row: (row.partition, row.file_name))


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _phash64(image: np.ndarray) -> int:
    from PIL import Image

    gray = np.asarray(Image.fromarray(image.astype(np.uint8)).convert("L").resize((32, 32)), dtype=np.float32)
    try:
        import cv2

        coeff = cv2.dct(gray)[:8, :8]
    except ImportError:
        coeff = np.fft.fft2(gray).real[:8, :8]
    flat = coeff.reshape(-1)
    median = float(np.median(flat[1:]))
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def compare_last_frames(reference: np.ndarray, decoded: np.ndarray) -> dict[str, float | int | bool]:
    from PIL import Image

    if reference.shape != decoded.shape:
        decoded = np.asarray(Image.fromarray(decoded.astype(np.uint8)).resize((reference.shape[1], reference.shape[0])))
    ref = reference.astype(np.float64) / 255.0
    dec = decoded.astype(np.float64) / 255.0
    mse = float(np.mean((ref - dec) ** 2))
    mae = float(np.mean(np.abs(ref - dec)))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))
    try:
        from skimage.metrics import structural_similarity

        ssim = float(structural_similarity(ref, dec, channel_axis=2, data_range=1.0))
    except ImportError:
        mu_x, mu_y = float(ref.mean()), float(dec.mean())
        var_x, var_y = float(ref.var()), float(dec.var())
        covariance = float(((ref - mu_x) * (dec - mu_y)).mean())
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    # bin().count() keeps the manifest audit compatible with the project's
    # Python 3.9 runtime while computing the same unsigned Hamming distance.
    phash_distance = bin(_phash64(reference) ^ _phash64(decoded)).count("1")
    passed = ssim >= 0.90 and psnr >= 20.0 and mae <= 0.08 and phash_distance <= 16
    return {"ssim": ssim, "psnr": psnr, "normalized_mae": mae, "phash_distance": phash_distance, "pass": passed}


def assert_no_partition_leakage(rows: Sequence[dict[str, Any] | TIDAClipRecord]) -> None:
    normalized = [row.to_dict() if isinstance(row, TIDAClipRecord) else row for row in rows]
    for key in ("source_video_id", "clip_sha256"):
        owners: dict[str, str] = {}
        for row in normalized:
            raw = str(row.get(key, ""))
            value = normalize_source_id(raw) if key == "source_video_id" else raw
            if not value:
                continue
            partition = str(row["partition"])
            if value in owners and owners[value] != partition:
                raise ValueError(f"partition leakage through {key}: {value}")
            owners[value] = partition
    for i, left in enumerate(normalized):
        for right in normalized[i + 1 :]:
            if left["partition"] == right["partition"]:
                continue
            if not left.get("endpoint_phash") or not right.get("endpoint_phash"):
                continue
            distance = (int(left["endpoint_phash"]) ^ int(right["endpoint_phash"])).bit_count()
            if distance <= 4 and abs(float(left.get("duration_seconds", 0)) - float(right.get("duration_seconds", 0))) <= 0.1 and abs(float(left.get("fps", 0)) - float(right.get("fps", 0))) <= 0.5:
                raise ValueError("partition leakage through endpoint near-duplicate")


def validate_records(records: Sequence[TIDAClipRecord], *, require_files: bool = True) -> dict[str, Any]:
    keys: set[tuple[str, str]] = set()
    errors: list[str] = []
    for record in records:
        key = (record.official_split, record.file_name)
        if key in keys:
            errors.append(f"duplicate key: {key}")
        keys.add(key)
        if require_files and (not record.target_image_path.is_file() or not record.clip_path.is_file()):
            errors.append(f"missing path: {record.file_name}")
    try:
        assert_no_partition_leakage(records)
    except ValueError as error:
        errors.append(str(error))
    return {"pass": not errors, "count": len(records), "errors": errors}


def load_manifest(path: str | Path) -> list[TIDAClipRecord]:
    return [TIDAClipRecord.from_dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_manifest(path: str | Path, records: Iterable[TIDAClipRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
