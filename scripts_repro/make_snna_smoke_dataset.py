import argparse
import json
import shutil
from pathlib import Path


def reset_dir(path: Path) -> None:
    path = path.resolve()
    if path.name != "dataset_smoke":
        raise RuntimeError(f"refusing to remove unexpected smoke root: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_bdd100k(repo: Path, out_root: Path, counts: dict) -> None:
    src_root = repo / "dataset" / "BDD100k" / "images" / "100k"
    dst_root = out_root / "BDD100k" / "images" / "100k"
    for split, count in counts.items():
        src_dir = src_root / split
        dst_dir = dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(src_dir.glob("*.jpg"))[:count]
        if len(images) < count:
            raise RuntimeError(f"BDD100K {split}: expected {count}, found {len(images)}")
        for image in images:
            shutil.copy2(image, dst_dir / image.name)


def copy_bdd_oia(repo: Path, out_root: Path, counts: dict) -> None:
    src_root = repo / "dataset" / "BDD-OIA"
    dst_root = out_root / "BDD-OIA"
    for split, count in counts.items():
        src_dir = src_root / split
        dst_dir = dst_root / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        labels = json.loads((src_root / f"{split}.json").read_text())
        selected = []
        for image in sorted(src_dir.glob("*.jpg")):
            if image.name in labels:
                selected.append(image)
            if len(selected) >= count:
                break
        if len(selected) < count:
            raise RuntimeError(f"BDD-OIA {split}: expected {count}, found {len(selected)}")
        subset_labels = {}
        for image in selected:
            shutil.copy2(image, dst_dir / image.name)
            subset_labels[image.name] = labels[image.name]
        (dst_root / f"{split}.json").write_text(json.dumps(subset_labels, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="SNNA repo root")
    parser.add_argument("--output", default="dataset_smoke", help="smoke dataset root under repo")
    parser.add_argument("--bdd100k_train", type=int, default=96)
    parser.add_argument("--bdd100k_val", type=int, default=16)
    parser.add_argument("--bdd100k_test", type=int, default=16)
    parser.add_argument("--bdd_oia_train", type=int, default=64)
    parser.add_argument("--bdd_oia_val", type=int, default=32)
    parser.add_argument("--bdd_oia_test", type=int, default=32)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out_root = (repo / args.output).resolve()
    reset_dir(out_root)
    copy_bdd100k(
        repo,
        out_root,
        {"train": args.bdd100k_train, "val": args.bdd100k_val, "test": args.bdd100k_test},
    )
    copy_bdd_oia(
        repo,
        out_root,
        {"train": args.bdd_oia_train, "val": args.bdd_oia_val, "test": args.bdd_oia_test},
    )
    print(json.dumps({
        "dataset_smoke": str(out_root),
        "bdd100k": {"train": args.bdd100k_train, "val": args.bdd100k_val, "test": args.bdd100k_test},
        "bdd_oia": {"train": args.bdd_oia_train, "val": args.bdd_oia_val, "test": args.bdd_oia_test},
    }, indent=2))


if __name__ == "__main__":
    main()
