from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch

from fate_oia.engine.dice_common import build_dice_model, load_config
from fate_oia.utils.dice_counterfactual_engine import DICECounterfactualEngine
from fate_oia.utils.dice_artifacts import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--base-checkpoint",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int,default=8); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); model=build_dice_model(cfg,a.base_checkpoint,device); image=torch.randn(a.batch_size,3,360,640,device=device)
    torch.cuda.reset_peak_memory_stats() if device.type=="cuda" else None; start=time.perf_counter()
    model.train(); target=torch.randint(0,2,(a.batch_size,4),device=device).float()
    with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
        output=model(image)
        engine=DICECounterfactualEngine(float(cfg["dice"]["license_temperature"]),int(cfg["counterfactual"]["max_actions_per_sample"]),
            float(cfg["counterfactual"]["batch_fraction"]),int(cfg["counterfactual"].get("topk_patches",64)))
        cf=engine.run(model,output,target,3)
        loss=output["action_logits_final"].float().square().mean()
        if cf["available"]: loss=loss+cf["directional_effect"].float().square().mean()
    loss.backward()
    if device.type=="cuda": torch.cuda.synchronize()
    payload={"batch_size":a.batch_size,"seconds":time.perf_counter()-start,"reserved_gb":torch.cuda.max_memory_reserved()/2**30 if device.type=="cuda" else 0,
             "allocated_gb":torch.cuda.max_memory_allocated()/2**30 if device.type=="cuda" else 0,"training_backward":True,
             "counterfactual_events":len(cf["atom_index"]) if cf["available"] else 0,
             "pass":device.type!="cuda" or torch.cuda.max_memory_reserved()/2**30<=float(cfg["runtime"]["max_reserved_memory_gb"])}
    write_json(Path(a.output_dir)/"DICE_PROFILE.json",payload); print(payload)


if __name__=="__main__": main()
