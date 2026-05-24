import argparse
import json
from pathlib import Path


def list_jpgs(root):
    return {p.name: str(p) for p in Path(root).rglob("*.jpg")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--output", default="repro_runs/oia_bdd100k_overlap_audit.json")
    args = ap.parse_args()

    repo = Path(args.repo)
    bdd100k_root = repo / "dataset" / "BDD100k" / "images" / "100k"
    bdd_oia_root = repo / "dataset" / "BDD-OIA"

    bdd100k_by_split = {}
    bdd100k_all = {}
    for split in ["train", "val", "test"]:
        files = list_jpgs(bdd100k_root / split)
        bdd100k_by_split[split] = files
        bdd100k_all.update(files)

    result = {
        "bdd100k_counts": {k: len(v) for k, v in bdd100k_by_split.items()},
        "bdd_oia": {},
        "overall": {},
        "examples": {},
    }
    all_oia = set()
    matched = set()
    unmatched = set()
    for split in ["train", "val", "test"]:
        oia_files = sorted(p.name for p in (bdd_oia_root / split).glob("*.jpg"))
        oia_set = set(oia_files)
        all_oia.update(oia_set)
        split_matched = oia_set & set(bdd100k_all)
        split_unmatched = oia_set - set(bdd100k_all)
        matched |= split_matched
        unmatched |= split_unmatched
        result["bdd_oia"][split] = {
            "count": len(oia_set),
            "matched_any_bdd100k": len(split_matched),
            "unmatched_any_bdd100k": len(split_unmatched),
            "match_rate": len(split_matched) / max(1, len(oia_set)),
            "matched_by_bdd100k_split": {
                bdd_split: len(oia_set & set(files))
                for bdd_split, files in bdd100k_by_split.items()
            },
        }
        result["examples"][f"{split}_matched"] = sorted(split_matched)[:10]
        result["examples"][f"{split}_unmatched"] = sorted(split_unmatched)[:10]

    result["overall"] = {
        "oia_unique_images": len(all_oia),
        "matched_unique": len(matched),
        "unmatched_unique": len(unmatched),
        "match_rate": len(matched) / max(1, len(all_oia)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
