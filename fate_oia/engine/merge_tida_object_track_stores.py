from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.datasets.tida_clip_manifest import load_manifest


def merge_track_stores(manifest_path: str | Path, store_paths: list[str | Path]) -> dict[str, object]:
    records = load_manifest(manifest_path)
    by_name: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    duplicates = 0
    store_rows: dict[str, int] = {}
    for raw_path in store_paths:
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        names = [str(value) for value in payload["file_names"]]
        tracks = payload["tracks_xy"]
        visibility = payload["visibility"]
        store_rows[str(path.resolve())] = len(names)
        for index, name in enumerate(names):
            key = name.lower()
            if key in by_name:
                duplicates += 1
                continue
            by_name[key] = (tracks[index], visibility[index], str(path.resolve()))
    missing = [row.file_name for row in records if row.file_name.lower() not in by_name]
    if missing:
        raise RuntimeError(f"track stores are missing {len(missing)} manifest rows: {missing[:5]}")
    ordered = [by_name[row.file_name.lower()] for row in records]
    return {
        "schema": "tida_frozen_cotracker_coordinates_v1",
        "manifest": str(Path(manifest_path).resolve()),
        "file_names": [row.file_name for row in records],
        "tracks_xy": torch.stack([item[0] for item in ordered]).contiguous(),
        "visibility": torch.stack([item[1] for item in ordered]).contiguous(),
        "audit": {
            "pass": True,
            "manifest_rows": len(records),
            "output_rows": len(ordered),
            "duplicate_candidates_ignored": duplicates,
            "store_rows": store_rows,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--store", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    payload = merge_track_stores(args.manifest, args.store)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    audit = payload["audit"]
    Path(args.audit_output).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
