import argparse
import os
import shutil
from pathlib import Path


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--output", default="dataset_fast/BDD-OIA-DINO/all_splits")
    ap.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated BDD-OIA image splits to include for self-supervised DINO.",
    )
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    if not splits:
        raise RuntimeError("No BDD-OIA splits were requested.")
    out_root = repo / args.output
    out_class = out_root / "train"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_class.mkdir(parents=True, exist_ok=True)

    images = []
    per_split = {}
    for split in splits:
        src_dir = repo / "dataset" / "BDD-OIA" / split
        split_images = sorted(src_dir.glob("*.jpg"))
        if not split_images:
            raise RuntimeError(f"No images under {src_dir}")
        per_split[split] = len(split_images)
        images.extend(split_images)
    if not images:
        raise RuntimeError("No BDD-OIA images found.")

    seen_names = set()
    for image in images:
        if image.name in seen_names:
            raise RuntimeError(f"Duplicate image filename across requested splits: {image.name}")
        seen_names.add(image.name)
        link_or_copy(image, out_class / image.name)
    print({
        "source_root": str(repo / "dataset" / "BDD-OIA"),
        "splits": splits,
        "per_split": per_split,
        "output_imagefolder_root": str(out_root),
        "class_subdir": "train",
        "image_count": len(images),
    })


if __name__ == "__main__":
    main()
