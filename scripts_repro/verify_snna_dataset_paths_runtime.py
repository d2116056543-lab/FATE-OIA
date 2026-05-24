#!/usr/bin/env python3
"""Runtime dataset-path verification using SNNA's actual loaders."""

import json
import os
import sys
from pathlib import Path

from torchvision import datasets, transforms


def main() -> None:
    root = Path("/mnt/e/sbw/SNNA_repro/SNNA")
    sys.path.insert(0, str(root))
    from multi_label_train import myDataset

    print("repo", root)

    bdd = root / "dataset/BDD100k/images/100k"
    dino_ds = datasets.ImageFolder(str(bdd))
    print("bdd100k_imagefolder_root", bdd)
    print("bdd100k_imagefolder_len", len(dino_ds))
    print("bdd100k_class_to_idx", dino_ds.class_to_idx)
    for split in ["train", "val", "test"]:
        p = bdd / split
        print(
            "bdd100k_split",
            split,
            "exists",
            p.exists(),
            "jpg_count",
            len(list(p.glob("*.jpg"))),
        )

    oia = root / "dataset/BDD-OIA"
    for split in ["train", "val", "test"]:
        split_dir = oia / split
        label_path = oia / f"{split}.json"
        labels = json.load(open(label_path, "r", encoding="utf-8"))
        jpgs = [x for x in os.listdir(split_dir) if x.endswith(".jpg")]
        bad = [x for x in jpgs[:1000] if x not in labels]
        print(
            "bdd_oia_split",
            split,
            "images",
            len(jpgs),
            "labels",
            len(labels),
            "first1000_missing_labels",
            len(bad),
        )
        repo_ds = myDataset(str(split_dir), str(label_path), transform=transforms.ToTensor())
        print("bdd_oia_repo_dataset_len", split, len(repo_ds))

    print("snna_path_verification_ok")


if __name__ == "__main__":
    main()
