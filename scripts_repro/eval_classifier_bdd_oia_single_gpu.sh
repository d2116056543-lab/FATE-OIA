#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/e/sbw/SNNA_repro/SNNA}"
ENV_NAME="${2:-SNNA}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$REPO_ROOT"

python multi_label_train.py \
  --num_labels 4 \
  --patch_size 8 \
  --batch_size_per_gpu 4 \
  --pretrained_weights ckp/backbone_200.pth \
  --data_path dataset/BDD-OIA \
  --output_dir ckp/classifier_bdd_oia \
  --evaluate
