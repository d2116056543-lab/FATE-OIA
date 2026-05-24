#!/usr/bin/env python3
"""Verify SNNA reproduction folders without requiring full BDD100K extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def count_jpgs(path: Path) -> int:
    return sum(1 for _ in path.glob("*.jpg")) if path.exists() else 0


def check_bdd_oia(root: Path) -> Dict[str, object]:
    out: Dict[str, object] = {"root": str(root), "exists": root.exists(), "splits": {}}
    for split in ("train", "val", "test"):
        split_dir = root / split
        json_path = root / f"{split}.json"
        split_info: Dict[str, object] = {
            "image_dir_exists": split_dir.exists(),
            "json_exists": json_path.exists(),
            "image_count": count_jpgs(split_dir),
        }
        if json_path.exists():
            labels = json.load(json_path.open("r", encoding="utf-8"))
            split_info["label_count"] = len(labels)
            split_info["sample"] = next(iter(labels.items())) if labels else None
            split_info["image_label_count_match"] = split_info["image_count"] == len(labels)
            bad = [k for k, v in list(labels.items())[:100] if not isinstance(v, list) or len(v) != 4]
            split_info["first_100_bad_label_rows"] = len(bad)
        out["splits"][split] = split_info
    return out


def check_bdd100k(root: Path) -> Dict[str, object]:
    image_root = root / "images" / "100k"
    out: Dict[str, object] = {
        "root": str(root),
        "image_root": str(image_root),
        "image_root_exists": image_root.exists(),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        out["splits"][split] = {
            "exists": (image_root / split).exists(),
            "sample_jpg_count_limited": len(list((image_root / split).glob("*.jpg"))) if (image_root / split).exists() else 0,
        }
    return out


def check_weights(root: Path) -> Dict[str, object]:
    names = [
        "classifier.pth.tar",
        "dino_deitsmall8_pretrain.pth",
        "dino_deitsmall8_linearweights.pth",
        "backbone_200.pth",
    ]
    out: Dict[str, object] = {}
    for name in names:
        path = root / name
        out[name] = {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
    ref = root / "reference"
    if ref.exists():
        for path in ref.glob("*.pth"):
            out[f"reference/{path.name}"] = {"exists": True, "bytes": path.stat().st_size}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    report = {
        "repo_root": str(repo),
        "readme_exists": (repo / "README.MD").exists(),
        "env_file_exists": (repo / "SNNA.yml").exists(),
        "scripts_repro_exists": (repo / "scripts_repro").exists(),
        "bdd_oia": check_bdd_oia(repo / "dataset" / "BDD-OIA"),
        "bdd100k": check_bdd100k(repo / "dataset" / "BDD100k"),
        "weights": check_weights(repo / "ckp"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
