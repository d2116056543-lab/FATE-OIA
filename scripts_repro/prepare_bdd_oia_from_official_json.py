#!/usr/bin/env python3
"""Prepare BDD-OIA in the file layout expected by Hongbo-Z/SNNA.

SNNA's `multi_label_train.py` expects:

dataset/BDD-OIA/
  train/*.jpg
  val/*.jpg
  test/*.jpg
  train.json
  val.json
  test.json

The public BDD-OIA archive commonly extracts to one flat `data/` image folder
plus COCO-style `*_25k_images_actions.json` files. This script creates the
SNNA layout without changing labels: it uses the first four action bits
`forward, stop, left, right`, matching SNNA's `--num_labels 4`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


SPLITS = ("train", "val", "test")
ACTION_JSON = {
    "train": "train_25k_images_actions.json",
    "val": "val_25k_images_actions.json",
    "test": "test_25k_images_actions.json",
}


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    if mode in {"hardlink", "auto"}:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    if mode in {"symlink", "auto"}:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(src, dst)
    return "copy"


def _load_split(src_root: Path, split: str) -> Tuple[Dict[str, List[int]], List[str], List[str]]:
    json_path = src_root / ACTION_JSON[split]
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    if len(images) != len(annotations):
        raise ValueError(
            f"{json_path} has {len(images)} images but {len(annotations)} annotations"
        )

    labels: Dict[str, List[int]] = {}
    image_names: List[str] = []
    bad: List[str] = []
    for image, ann in zip(images, annotations):
        name = image.get("file_name")
        category = ann.get("category")
        if not name or not isinstance(category, list) or len(category) < 4:
            bad.append(str(image))
            continue
        labels[name] = [int(x) for x in category[:4]]
        image_names.append(name)
    return labels, image_names, bad


def prepare(src_root: Path, out_root: Path, link_mode: str) -> Dict[str, object]:
    src_root = src_root.resolve()
    out_root = out_root.resolve()
    image_root = src_root / "data"
    if not image_root.exists():
        raise FileNotFoundError(f"Missing BDD-OIA flat image directory: {image_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {
        "source_root": str(src_root),
        "output_root": str(out_root),
        "label_order": ["forward", "stop", "left", "right"],
        "ignored_source_label": "confuse",
        "splits": {},
    }

    for split in SPLITS:
        labels, image_names, bad_records = _load_split(src_root, split)
        split_dir = out_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        missing: List[str] = []
        link_stats: Dict[str, int] = {}
        for name in image_names:
            src = image_root / name
            dst = split_dir / name
            if not src.exists():
                missing.append(name)
                continue
            method = _link_or_copy(src, dst, link_mode)
            link_stats[method] = link_stats.get(method, 0) + 1

        with (out_root / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(labels, f)

        summary["splits"][split] = {
            "json": str(src_root / ACTION_JSON[split]),
            "images_in_json": len(image_names),
            "labels_written": len(labels),
            "bad_records": len(bad_records),
            "missing_images": len(missing),
            "link_stats": link_stats,
            "sample_missing_images": missing[:20],
        }
        if missing:
            raise FileNotFoundError(
                f"{split}: {len(missing)} images referenced by JSON are missing. "
                f"First missing: {missing[:5]}"
            )

    with (out_root / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument(
        "--link_mode",
        default="auto",
        choices=["auto", "hardlink", "symlink", "copy"],
        help="auto tries hardlink, then symlink, then copy.",
    )
    args = parser.parse_args()
    summary = prepare(args.source_root, args.output_root, args.link_mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
