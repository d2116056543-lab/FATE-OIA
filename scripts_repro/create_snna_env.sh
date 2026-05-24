#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/e/sbw/SNNA_repro/SNNA}"
ENV_NAME="${2:-SNNA}"

source /opt/conda/etc/profile.d/conda.sh
cd "$REPO_ROOT"

grep -v '^prefix:' SNNA.yml > SNNA_no_prefix.yml
echo "Wrote SNNA_no_prefix.yml"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda_env_exists name=$ENV_NAME"
else
  conda env create -n "$ENV_NAME" -f SNNA_no_prefix.yml
fi

conda activate "$ENV_NAME"
python - <<'PY'
import torch, torchvision
print("snna_env_python_ok")
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
