#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate SNNA

REPO="${REPO:-/mnt/e/sbw/SNNA_repro/SNNA}"
RUN_ROOT="${RUN_ROOT:-$REPO/repro_runs/oia_fast_$(date +%Y%m%d_%H%M%S)}"
DINO_BATCH="${DINO_BATCH:-8}"
CLASSIFIER_BATCH="${CLASSIFIER_BATCH:-4}"
DINO_EPOCHS="${DINO_EPOCHS:-50}"
CLASSIFIER_EPOCHS="${CLASSIFIER_EPOCHS:-100}"
DINO_SPLITS="${DINO_SPLITS:-train,val,test}"
DINO_DATASET_TAG="${DINO_DATASET_TAG:-all_splits}"
DINO_RESUME_CHECKPOINT="${DINO_RESUME_CHECKPOINT:-}"
MASTER_PORT="${MASTER_PORT:-29661}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{dino,classifier,logs}
export PYTHONUNBUFFERED=1
export SNNA_DIST_BACKEND="${SNNA_DIST_BACKEND:-gloo}"

python scripts_repro/build_oia_train_dino_dataset.py \
  --repo "$REPO" \
  --output "dataset_fast/BDD-OIA-DINO/$DINO_DATASET_TAG" \
  --splits "$DINO_SPLITS" \
  2>&1 | tee "$RUN_ROOT/logs/build_oia_train_dino_dataset.log"

DINO_DATA_PATH="$REPO/dataset_fast/BDD-OIA-DINO/$DINO_DATASET_TAG"

if [[ -n "$DINO_RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$DINO_RESUME_CHECKPOINT" ]]; then
    echo "missing DINO_RESUME_CHECKPOINT=$DINO_RESUME_CHECKPOINT" >&2
    exit 2
  fi
  cp "$DINO_RESUME_CHECKPOINT" "$RUN_ROOT/dino/checkpoint.pth"
  python - <<PY
import torch, json
p = "$RUN_ROOT/dino/checkpoint.pth"
ck = torch.load(p, map_location="cpu")
print(json.dumps({"resume_checkpoint": "$DINO_RESUME_CHECKPOINT", "resume_epoch": ck.get("epoch"), "keys": sorted(list(ck.keys()))}))
PY
fi

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
  "mode": "oia_dino_selfsupervised_then_classifier",
  "repo": "$REPO",
  "dino_data_path": "$DINO_DATA_PATH",
  "dino_splits": "$DINO_SPLITS",
  "classifier_data_path": "$REPO/dataset/BDD-OIA",
  "dino_batch_size_per_gpu": $DINO_BATCH,
  "classifier_batch_size_per_gpu": $CLASSIFIER_BATCH,
  "dino_epochs": $DINO_EPOCHS,
  "classifier_epochs": $CLASSIFIER_EPOCHS,
  "dino_resume_checkpoint": "$DINO_RESUME_CHECKPOINT",
  "note": "DINO self-supervised on requested BDD-OIA image splits, then original supervised BDD-OIA classifier with official split files."
}
JSON

echo "snna_oia_fast_stage=dino epochs=$DINO_EPOCHS batch=$DINO_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$MASTER_PORT" main_dino.py \
  --patch_size 8 \
  --batch_size_per_gpu "$DINO_BATCH" \
  --epochs "$DINO_EPOCHS" \
  --saveckp_freq 10 \
  --num_workers 8 \
  --data_path "$DINO_DATA_PATH" \
  --output_dir "$RUN_ROOT/dino" \
  2>&1 | tee "$RUN_ROOT/logs/dino_fast.log"

python scripts_repro/export_backbone_200.py \
  --dino_checkpoint "$RUN_ROOT/dino/checkpoint.pth" \
  --output_path "$RUN_ROOT/backbone_oia_fast.pth" \
  --force \
  2>&1 | tee "$RUN_ROOT/logs/export_backbone.log"

echo "snna_oia_fast_stage=classifier epochs=$CLASSIFIER_EPOCHS batch=$CLASSIFIER_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$((MASTER_PORT + 1))" multi_label_train.py \
  --num_labels 4 \
  --patch_size 8 \
  --batch_size_per_gpu "$CLASSIFIER_BATCH" \
  --epochs "$CLASSIFIER_EPOCHS" \
  --num_workers 8 \
  --pretrained_weights "$RUN_ROOT/backbone_oia_fast.pth" \
  --data_path "$REPO/dataset/BDD-OIA" \
  --output_dir "$RUN_ROOT/classifier" \
  2>&1 | tee "$RUN_ROOT/logs/classifier_fast.log"

echo "snna_oia_fast_pipeline_done run_root=$RUN_ROOT"
