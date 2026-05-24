#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate SNNA

REPO="${REPO:-/mnt/e/sbw/SNNA_repro/SNNA}"
RUN_ROOT="${RUN_ROOT:-$REPO/repro_runs/smoke_$(date +%Y%m%d_%H%M%S)}"
DINO_BATCH="${DINO_BATCH:-16}"
CLASSIFIER_BATCH="${CLASSIFIER_BATCH:-4}"
MASTER_PORT="${MASTER_PORT:-29621}"

cd "$REPO"
mkdir -p "$RUN_ROOT"/{dino,classifier,logs}
export PYTHONUNBUFFERED=1
export SNNA_DIST_BACKEND="${SNNA_DIST_BACKEND:-gloo}"

python scripts_repro/make_snna_smoke_dataset.py --repo "$REPO" --output dataset_smoke | tee "$RUN_ROOT/logs/make_smoke_dataset.log"

(
  echo "timestamp,memory.used [MiB],memory.total [MiB],utilization.gpu [%]"
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 2
  done
) > "$RUN_ROOT/logs/vram_monitor.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" >/dev/null 2>&1 || true' EXIT

echo "snna_smoke_stage=dino batch=$DINO_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$MASTER_PORT" main_dino.py \
  --patch_size 8 \
  --batch_size_per_gpu "$DINO_BATCH" \
  --epochs 1 \
  --warmup_epochs 0 \
  --saveckp_freq 1 \
  --num_workers 4 \
  --data_path "$REPO/dataset_smoke/BDD100k/images/100k" \
  --output_dir "$RUN_ROOT/dino" \
  2>&1 | tee "$RUN_ROOT/logs/dino_smoke.log"

python scripts_repro/export_backbone_200.py \
  --dino_checkpoint "$RUN_ROOT/dino/checkpoint.pth" \
  --output_path "$RUN_ROOT/backbone_200.pth" \
  --force \
  2>&1 | tee "$RUN_ROOT/logs/export_backbone.log"

echo "snna_smoke_stage=classifier batch=$CLASSIFIER_BATCH run_root=$RUN_ROOT"
python -m torch.distributed.launch --nproc_per_node=1 --master_port "$((MASTER_PORT + 1))" multi_label_train.py \
  --num_labels 4 \
  --patch_size 8 \
  --batch_size_per_gpu "$CLASSIFIER_BATCH" \
  --epochs 1 \
  --num_workers 4 \
  --pretrained_weights "$RUN_ROOT/backbone_200.pth" \
  --data_path "$REPO/dataset_smoke/BDD-OIA" \
  --output_dir "$RUN_ROOT/classifier" \
  2>&1 | tee "$RUN_ROOT/logs/classifier_smoke.log"

python - <<'PY'
from pathlib import Path
import csv, json, os
run_root = Path(os.environ["RUN_ROOT"])
vram = run_root / "logs" / "vram_monitor.csv"
max_used = 0
if vram.exists():
    for row in csv.reader(vram.open()):
        if row and row[0] != "timestamp":
            try:
                max_used = max(max_used, int(row[1].strip()))
            except Exception:
                pass
summary = {
    "run_root": str(run_root),
    "dino_checkpoint_exists": (run_root / "dino" / "checkpoint.pth").exists(),
    "exported_backbone_exists": (run_root / "backbone_200.pth").exists(),
    "classifier_checkpoint_exists": (run_root / "classifier" / "360_checkpoint.pth.tar").exists(),
    "max_vram_mib": max_used,
}
(run_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
