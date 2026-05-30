from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_oia.datasets.bdd100k_grounding import bdd_oia_base_stem, load_bdd100k_objects
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.utils.cafe_artifacts import write_json


def load_grounding_cache_jsonl(path: str | Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return cache
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            fn = str(rec.get("file_name") or rec.get("image") or rec.get("image_path") or "")
            if fn:
                for key in _candidate_keys(fn):
                    cache.setdefault(key, rec)
            for extra in ("image_path", "label_json"):
                value = rec.get(extra)
                if isinstance(value, str):
                    for key in _candidate_keys(value):
                        cache.setdefault(key, rec)
    return cache


def _candidate_keys(file_name: str) -> list[str]:
    text = str(file_name).replace("\\", "/")
    p = Path(text)
    raw = [text, p.name, p.stem, bdd_oia_base_stem(text)]
    out: list[str] = []
    for item in raw:
        if item and item not in out:
            out.append(item)
        low = item.lower()
        if low and low not in out:
            out.append(low)
    return out


def resolve_grounding_record(file_name: str, grounding_cache: dict[str, dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str]:
    modes = ["exact", "basename", "stem", "base_stem"]
    candidates = _candidate_keys(file_name)
    for idx, key in enumerate(candidates):
        if key in grounding_cache:
            return grounding_cache[key], key, modes[min(idx // 2, len(modes) - 1)]
    return None, "", "missing"


def _path_from_record(record: dict[str, Any], key: str) -> str | None:
    value = record.get(key)
    if isinstance(value, str):
        return value
    paths = record.get("paths")
    if isinstance(paths, dict) and isinstance(paths.get(key), str):
        return paths[key]
    return None


def objects_from_record(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not record:
        return []
    objects = record.get("objects")
    if isinstance(objects, list):
        return [o for o in objects if isinstance(o, dict)]
    label_json = record.get("label_json")
    if isinstance(label_json, dict):
        out: list[dict[str, Any]] = []
        for frame in label_json.get("frames", []) or []:
            if isinstance(frame, dict):
                out.extend([o for o in frame.get("objects", []) or [] if isinstance(o, dict)])
                out.extend([o for o in frame.get("labels", []) or [] if isinstance(o, dict)])
        return out
    label_path = _path_from_record(record, "label_json")
    if label_path and Path(label_path).exists():
        try:
            return load_bdd100k_objects(label_path)
        except Exception:
            return []
    return []


def count_record_evidence(record: dict[str, Any] | None) -> dict[str, int]:
    objects = objects_from_record(record)
    counts = {
        "object_box_rows": 0,
        "object_poly_rows": 0,
        "lane_poly_rows": 0,
        "drivable_map_rows": 0,
        "drivable_area_rows": 0,
    }
    if not record:
        return counts
    for obj in objects:
        cat = str(obj.get("category", ""))
        if isinstance(obj.get("box2d"), dict):
            counts["object_box_rows"] += 1
        if obj.get("poly2d"):
            if cat.startswith("lane/"):
                counts["lane_poly_rows"] += 1
            elif cat.startswith("area/"):
                counts["drivable_area_rows"] += 1
            else:
                counts["object_poly_rows"] += 1
    drive_path = _path_from_record(record, "drivable_map")
    if drive_path and Path(drive_path).exists():
        counts["drivable_map_rows"] += 1
    return counts


def audit_split(
    split: str,
    data_root: str,
    raw_root: str,
    grounding_cache: dict[str, dict[str, Any]],
    max_samples: int = 0,
) -> dict[str, Any]:
    dataset = BDDOIAMultiTaskDataset(data_root=data_root, raw_root=raw_root, split=split, load_image=False)
    samples = dataset.samples[: max_samples or None]
    stats = {
        "split": split,
        "total": len(samples),
        "exact_hits": 0,
        "basename_hits": 0,
        "stem_hits": 0,
        "base_stem_hits": 0,
        "missing": 0,
        "fallback_only_rows": 0,
        "examples": [],
        "object_box_rows": 0,
        "object_poly_rows": 0,
        "lane_poly_rows": 0,
        "drivable_map_rows": 0,
        "drivable_area_rows": 0,
    }
    for sample in samples:
        rec, key, mode = resolve_grounding_record(sample.file_name, grounding_cache)
        if rec is None:
            stats["missing"] += 1
            continue
        stats[f"{mode}_hits"] = int(stats.get(f"{mode}_hits", 0)) + 1
        counts = count_record_evidence(rec)
        for k, v in counts.items():
            stats[k] += int(v)
        if sum(counts.values()) == 0:
            stats["fallback_only_rows"] += 1
        if len(stats["examples"]) < 5:
            stats["examples"].append({"file_name": sample.file_name, "matched_key": key, "match_mode": mode, **counts})
    hits = stats["total"] - stats["missing"]
    stats["key_hit_rate"] = hits / max(1, stats["total"])
    if stats["lane_poly_rows"] == 0 and stats["drivable_map_rows"] == 0 and stats["drivable_area_rows"] == 0:
        stats["lane_drivable_note"] = "lane/drivable evidence unavailable in this cache; object evidence is primary."
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--grounding_cache_jsonl", default=".background_runs/fate_oia_grounding_cache_20260525.jsonl")
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--output_dir", default=".background_runs/cafe_oia_v2_preflight")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = load_grounding_cache_jsonl(args.grounding_cache_jsonl)
    rows = {split: audit_split(split, args.data_root, args.raw_root, cache, args.max_samples) for split in ("train", "val", "test")}
    for split, stats in rows.items():
        write_json(out / f"evidence_audit_{split}.json", stats)
    merged = {"cache_path": args.grounding_cache_jsonl, "splits": rows}
    write_json(out / "evidence_audit_real_split.json", merged)
    print(json.dumps(merged, ensure_ascii=False))


if __name__ == "__main__":
    main()
