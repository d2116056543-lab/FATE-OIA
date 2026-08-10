from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.engine.vetra_common import build_vetra_model, load_config, make_loader
from fate_oia.utils.vetra_artifacts import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); dataset=make_dataset(cfg,"train"); rows=[]
    for batch_size in (8,6,4):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); model=build_vetra_model(cfg,a.source_checkpoint,device)
            batch=next(iter(make_loader(Subset(dataset,list(range(batch_size))),batch_size,False,0,cfg))); started=time.perf_counter()
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                out=model(batch["image"].to(device),alpha=.1); loss=out["action_logits_final"].square().mean()
            loss.backward(); torch.cuda.synchronize(); seconds=time.perf_counter()-started
            rows.append({"batch_size":batch_size,"pass":True,"seconds":seconds,"samples_per_second":batch_size/seconds,
                         "peak_reserved_gb":torch.cuda.max_memory_reserved()/2**30})
            del model
        except torch.cuda.OutOfMemoryError:
            rows.append({"batch_size":batch_size,"pass":False,"reason":"oom"})
    valid=[r for r in rows if r["pass"] and r["peak_reserved_gb"]<=float(cfg["runtime"]["max_reserved_memory_gb"])]
    best=max(valid,key=lambda r:r["samples_per_second"]) if valid else None
    payload={"pass":best is not None,"candidates":rows,"recommended_batch_size":best["batch_size"] if best else None}
    write_json(Path(a.output_dir)/"VETRA_RUNTIME_PROFILE.json",payload); print(json.dumps(payload,indent=2))
    if not payload["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
