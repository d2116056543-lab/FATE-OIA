#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/e/sbw/SNNA_repro/SNNA}"
ENV_NAME="${2:-SNNA}"

source /opt/conda/etc/profile.d/conda.sh
cd "$REPO_ROOT"

/opt/conda/bin/python - <<'PY'
from pathlib import Path
src = Path("SNNA.yml")
dst = Path("SNNA_tuna_no_prefix.yml")
lines = []
in_channels = False
for line in src.read_text().splitlines():
    if line.startswith("prefix:"):
        continue
    if line == "channels:":
        lines.append(line)
        lines.append("  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch")
        lines.append("  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main")
        lines.append("  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free")
        in_channels = True
        continue
    if in_channels:
        if line.startswith("  - "):
            continue
        in_channels = False
    lines.append(line)
dst.write_text("\n".join(lines) + "\n")
print(f"Wrote {dst}")
PY

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda_env_exists name=$ENV_NAME"
else
  conda env create -n "$ENV_NAME" -f SNNA_tuna_no_prefix.yml
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
