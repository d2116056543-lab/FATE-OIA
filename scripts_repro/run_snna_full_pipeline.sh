#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate SNNA

REPO="${REPO:-/mnt/e/sbw/SNNA_repro/SNNA}"
RUN_ROOT="${RUN_ROOT:-$REPO/repro_runs/full_$(date +%Y%m%d_%H%M%S)}"
DINO_BATCH="${DINO_BATCH:-16}"
CLASSIFIER_BATCH="${CLASSIFIER_BATCH:-4}"
MASTER_PORT="${MASTER_PORT:-29631}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{dino,classifier,logs}
export PYTHONUNBUFFERED=1
export SNNA_DIST_BACKEND="${SNNA_DIST_BACKEND:-gloo}"

(
  echo "timestamp,memory.used [MiB],memory.total [MiB],utilization.gpu [%]"
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 10
  done
) > "$RUN_ROOT/logs/vram_monitor.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" >/dev/null 2>&1 || true' EXIT

cat > "$RUN_ROOT/run_config.json" <<JSON
{
  "repo": "$REPO",
  "dino_batch_size_per_gpu": $DINO_BATCH,
  "classifier_batch_size_per_gpu": $CLASSIFIER_BATCH,
  "dino_epochs": 200,
  "classifier_epochs": 100,
  "bdd100k_data_path": "$REPO/dataset/BDD100k/images/100k",
  "bdd_oia_data_path": "$REPO/dataset/BDD-OIA",
  "paper_repo_alignment": "README command preserved: DINO on BDD100k/images/100k then BDD-OIA classifier with exported backbone_200.pth"
}
JSON

echo "snna_full_stage=dino batch=$DINO_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$MASTER_PORT" main_dino.py \
  --patch_size 8 \
  --batch_size_per_gpu "$DINO_BATCH" \
  --epochs 200 \
  --saveckp_freq 10 \
  --num_workers 8 \
  --data_path "$REPO/dataset/BDD100k/images/100k" \
  --output_dir "$RUN_ROOT/dino" \
  2>&1 | tee "$RUN_ROOT/logs/dino_full.log"

python scripts_repro/export_backbone_200.py \
  --dino_checkpoint "$RUN_ROOT/dino/checkpoint.pth" \
  --output_path "$RUN_ROOT/backbone_200.pth" \
  --force \
  2>&1 | tee "$RUN_ROOT/logs/export_backbone.log"

cp "$RUN_ROOT/backbone_200.pth" "$REPO/ckp/backbone_200.pth"

echo "snna_full_stage=classifier batch=$CLASSIFIER_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$((MASTER_PORT + 1))" multi_label_train.py \
  --num_labels 4 \
  --patch_size 8 \
  --batch_size_per_gpu "$CLASSIFIER_BATCH" \
  --epochs 100 \
  --num_workers 8 \
  --pretrained_weights "$RUN_ROOT/backbone_200.pth" \
  --data_path "$REPO/dataset/BDD-OIA" \
  --output_dir "$RUN_ROOT/classifier" \
  2>&1 | tee "$RUN_ROOT/logs/classifier_full.log"

echo "snna_full_pipeline_done run_root=$RUN_ROOT"
