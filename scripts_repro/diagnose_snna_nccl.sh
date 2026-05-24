#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate SNNA
cd /mnt/e/sbw/SNNA_repro/SNNA

echo "---NVIDIA SMI---"
nvidia-smi

echo "---TORCH CUDA BASIC---"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
print("tensor_sum", float((x @ x).sum().detach().cpu()))
PY

cat > /tmp/snna_nccl_probe.py <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
dist.barrier()
print("nccl_probe_ok", rank, world_size)
dist.destroy_process_group()
PY

echo "---NCCL PROBE DEFAULT---"
set +e
python -m torch.distributed.launch --nproc_per_node=1 --master_port 29641 /tmp/snna_nccl_probe.py
default_code=$?
set -e
echo "default_exit=$default_code"

echo "---NCCL PROBE WSL SAFE ENV---"
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29642 /tmp/snna_nccl_probe.py
