from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, load_bdd100k_objects


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--bdd100k_root", default="raw_data/BDD100K")
    ap.add_argument("--output", default="outputs/fate_oia_grounding_cache.jsonl")
    args = ap.parse_args()
    index = BDD100KGroundingIndex(args.bdd100k_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    with out.open("w", encoding="utf-8") as f:
        for split in ["train", "val", "test"]:
            ds = BDDOIAMultiTaskDataset(args.data_root, args.raw_root, split=split, action_dim=4)
            files = [s.file_name for s in ds.samples]
            summary[split] = index.audit_file_names(files)
            for s in ds.samples:
                paths = index.lookup(s.file_name)
                categories = []
                if paths.label_json:
                    try:
                        objects = load_bdd100k_objects(paths.label_json)
                        categories = sorted(set(str(o.get("category")) for o in objects if o.get("category")))
                    except Exception:
                        categories = []
                f.write(json.dumps({
                    "split": split,
                    "file_name": s.file_name,
                    "image_path": s.image_path,
                    "label_json": paths.label_json,
                    "drivable_map": paths.drivable_map,
                    "semantic_seg": paths.semantic_seg,
                    "categories": categories,
                }, ensure_ascii=False) + "\n")
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "summary": str(summary_path), "stats": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
