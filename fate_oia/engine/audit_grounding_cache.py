from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def audit_cache(path: str | Path) -> dict:
    path = Path(path)
    split = defaultdict(lambda: {"count": 0, "label_json": 0, "drivable_map": 0, "semantic_seg": 0})
    cats = Counter()
    object_counts = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            s = rec.get("split", "unknown")
            split[s]["count"] += 1
            for key in ["label_json", "drivable_map", "semantic_seg"]:
                if rec.get(key):
                    split[s][key] += 1
            rec_cats = rec.get("categories") or []
            object_counts.append(len(rec_cats))
            cats.update(str(c) for c in rec_cats)
    split_out = {}
    for s, row in split.items():
        n = max(row["count"], 1)
        split_out[s] = {
            **row,
            "label_json_coverage": row["label_json"] / n,
            "drivable_map_coverage": row["drivable_map"] / n,
            "semantic_seg_coverage": row["semantic_seg"] / n,
        }
    return {
        "cache": str(path),
        "splits": split_out,
        "category_counts": dict(cats.most_common()),
        "mean_unique_categories_per_sample": sum(object_counts) / max(len(object_counts), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit FATE-OIA grounding cache coverage and object category breakdown.")
    ap.add_argument("--cache_jsonl", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = audit_cache(args.cache_jsonl)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()