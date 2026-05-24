#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/e/sbw/SNNA_repro/SNNA}"
ENV_NAME="${2:-adapt}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

python - <<'PY'
from pathlib import Path
import json
from PIL import Image

root = Path("dataset/BDD-OIA")
assert root.exists(), root
for split in ["train", "val", "test"]:
    labels = json.load(open(root / f"{split}.json", "r"))
    img_dir = root / split
    jpgs = sorted(img_dir.glob("*.jpg"))
    print(split, "labels", len(labels), "jpgs", len(jpgs), "first", jpgs[0].name if jpgs else None)
    if jpgs:
        im = Image.open(jpgs[0])
        print(split, "sample_size", im.size, "label", labels[jpgs[0].name])

print("snna_dataset_smoke_ok")
PY
