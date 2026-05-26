from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_oia.datasets.bdd100k_grounding import load_bdd100k_objects
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.train_fate_oia import load_grounding_cache, load_reason_grounding_rules


def _has_box(obj: dict[str, Any]) -> bool:
    box = obj.get("box2d") or {}
    return all(k in box for k in ("x1", "y1", "x2", "y2"))


def _has_poly(obj: dict[str, Any]) -> bool:
    poly = obj.get("poly2d")
    return isinstance(poly, list) and len(poly) > 0


def _target_counts(objects: list[dict[str, Any]], categories: set[str], drivable_map: str | None) -> dict[str, int]:
    object_count = 0
    lane_count = 0
    drivable_count = 0
    for obj in objects:
        cat = str(obj.get("category", ""))
        if cat not in categories:
            continue
        if _has_box(obj):
            object_count += 1
        if cat.startswith("lane/") and _has_poly(obj):
            lane_count += 1
        if cat.startswith("area/") and _has_poly(obj):
            drivable_count += 1
    if drivable_map and any(c.startswith("area/") for c in categories):
        drivable_count += 1
    return {"object": object_count, "lane": lane_count, "drivable": drivable_count}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit reason-grounding category coverage for BDD-OIA/Bdd100K cache.")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--grounding_cache_jsonl", required=True)
    ap.add_argument("--reason_grounding_rules", default="configs/reason_grounding_rules.yaml")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dataset = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=args.split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=False,
    )
    cache = load_grounding_cache(args.grounding_cache_jsonl)
    rules = load_reason_grounding_rules(args.reason_grounding_rules, args.reason_dim)
    per_reason: dict[int, dict[str, Any]] = {
        idx: {
            "reason_idx": idx,
            "positive_count": 0,
            "mapped_categories": sorted(cats),
            "object_box_target_count": 0,
            "lane_poly_target_count": 0,
            "drivable_target_count": 0,
            "empty_target_count": 0,
            "missing_cache_count": 0,
        }
        for idx, cats in sorted(rules.items())
    }
    for sample in dataset.samples:
        rec = cache.get(sample.file_name)
        objects: list[dict[str, Any]] = []
        if rec and rec.get("label_json"):
            try:
                objects = load_bdd100k_objects(rec["label_json"])
            except Exception:
                objects = []
        reasons = list(sample.reason)
        for idx, cats in rules.items():
            if idx >= len(reasons) or float(reasons[idx]) <= 0:
                continue
            row = per_reason[idx]
            row["positive_count"] += 1
            if not rec:
                row["missing_cache_count"] += 1
                row["empty_target_count"] += 1
                continue
            counts = _target_counts(objects, cats, rec.get("drivable_map"))
            row["object_box_target_count"] += counts["object"]
            row["lane_poly_target_count"] += counts["lane"]
            row["drivable_target_count"] += counts["drivable"]
            if counts["object"] + counts["lane"] + counts["drivable"] == 0:
                row["empty_target_count"] += 1
    for row in per_reason.values():
        pos = max(1, int(row["positive_count"]))
        row["object_rate"] = row["object_box_target_count"] / pos
        row["lane_rate"] = row["lane_poly_target_count"] / pos
        row["drivable_rate"] = row["drivable_target_count"] / pos
        row["empty_rate"] = row["empty_target_count"] / pos

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": args.split,
        "sample_count": len(dataset.samples),
        "grounding_cache_jsonl": str(args.grounding_cache_jsonl),
        "reason_grounding_rules": str(args.reason_grounding_rules),
        "per_reason": list(per_reason.values()),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
