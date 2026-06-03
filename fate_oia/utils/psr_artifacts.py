from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: str | Path, max_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        remaining = max_bytes
        while True:
            size = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining)
            if size <= 0:
                break
            chunk = f.read(size)
            if not chunk:
                break
            h.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
    return h.hexdigest()


def torch_load(path: str | Path) -> torch.Tensor:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def torch_save(path: str | Path, tensor: torch.Tensor) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    torch.save(tensor.detach().cpu(), str(p))


def first_existing(base: str | Path, candidates: Iterable[str]) -> Path | None:
    b = Path(base)
    for rel in candidates:
        p = b / rel
        if p.exists():
            return p
    return None


def newest_matching(globs: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in globs:
        out.extend(Path().glob(pattern) if not Path(pattern).is_absolute() else _absolute_glob(pattern))
    return sorted(set(out), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)


def _absolute_glob(pattern: str) -> list[Path]:
    import glob

    return [Path(p) for p in glob.glob(pattern, recursive=True)]
