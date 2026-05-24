#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/e/sbw/SNNA_repro/SNNA}"
ENV_NAME="${2:-SNNA}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

mkdir -p ckp/dino_bdd100k

python -m torch.distributed.launch \
  --nproc_per_node=1 \
  main_dino.py \
  --patch_size 8 \
  --batch_size_per_gpu 16 \
  --epochs 200 \
  --saveckp_freq 10 \
  --data_path dataset/BDD100k/images/100k \
  --output_dir ckp/dino_bdd100k
