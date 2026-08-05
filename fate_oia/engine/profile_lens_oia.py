from __future__ import annotations

import argparse
import json
import time

import torch

from fate_oia.models.lens_oia_model import LENSOIAModel


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--device",default="cuda"); parser.add_argument("--batch-size",type=int,default=1); parser.add_argument("--use-mock-dino",action="store_true"); args=parser.parse_args()
    device=torch.device(args.device); model=LENSOIAModel(use_mock_dino=args.use_mock_dino).to(device).eval(); image=torch.randn(args.batch_size,3,360,640,device=device)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    start=time.perf_counter()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"): out=model(image,progress=1.0)
    if device.type=="cuda": torch.cuda.synchronize(device)
    print(json.dumps({"seconds":time.perf_counter()-start,"allocated_gb":torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=="cuda" else 0.0,"reserved_gb":torch.cuda.max_memory_reserved(device)/(1024**3) if device.type=="cuda" else 0.0,"shape":list(out["action_logits_final"].shape)}))


if __name__=="__main__": main()
