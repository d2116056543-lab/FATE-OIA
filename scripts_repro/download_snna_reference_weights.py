#!/usr/bin/env python3
"""Download reference DINO weights used to initialize SNNA reproduction.

The SNNA repo does not ship the BDD100K self-supervised checkpoint
`backbone_200.pth`; that checkpoint must be produced by running `main_dino.py`
for 200 epochs on BDD100K. The downloadable weights here are the public DINO
ViT-S/8 ImageNet-pretrained initialization and reference linear head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Dict


WEIGHTS = {
    "dino_deitsmall8_pretrain.pth": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
    "dino_deitsmall8_linearweights.pth": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path, min_bytes: int, force: bool = False) -> Dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size >= min_bytes and not force:
        return {
            "path": str(path),
            "url": url,
            "status": "exists",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    tmp = path.with_suffix(path.suffix + ".tmp")
    start = time.time()
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as f:
        total = response.headers.get("Content-Length")
        read = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if read % (50 * 1024 * 1024) < 1024 * 1024:
                print(f"download_progress path={path.name} bytes={read} total={total}", flush=True)
    tmp.replace(path)
    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"Downloaded {path} is too small: {size} bytes")
    return {
        "path": str(path),
        "url": url,
        "status": "downloaded",
        "bytes": size,
        "seconds": round(time.time() - start, 2),
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_bytes", type=int, default=1024 * 1024)
    args = parser.parse_args()

    results = {}
    for name, url in WEIGHTS.items():
        results[name] = download(url, args.output_dir / name, args.min_bytes, args.force)

    summary_path = args.output_dir / "download_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
