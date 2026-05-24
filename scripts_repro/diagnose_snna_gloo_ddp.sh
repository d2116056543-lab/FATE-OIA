#!/usr/bin/env bash
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate SNNA

cat > /tmp/snna_gloo_ddp_probe.py <<'PY'
import os
import torch
import torch.distributed as dist
import torch.nn as nn

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
dist.init_process_group(backend="gloo", init_method="env://", rank=rank, world_size=world_size)
torch.cuda.set_device(0)
model = nn.Linear(8, 4).cuda()
ddp = nn.parallel.DistributedDataParallel(model, device_ids=[0])
x = torch.randn(2, 8, device="cuda")
y = ddp(x).sum()
y.backward()
print("gloo_cuda_ddp_probe_ok", rank, world_size, float(y.detach().cpu()))
dist.destroy_process_group()
PY

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29643 /tmp/snna_gloo_ddp_probe.py
