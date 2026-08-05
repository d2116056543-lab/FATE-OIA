from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from fate_oia.models.lens_oia_model import LENSOIAModel


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",default="configs/fate_oia_train_360x640_lens_oia_v1.yaml"); parser.add_argument("--device",default="cuda"); parser.add_argument("--batch-size",type=int,default=6); parser.add_argument("--factor-chunk-size",type=int,default=21); parser.add_argument("--warmup",type=int,default=2); parser.add_argument("--steps",type=int,default=5); parser.add_argument("--output"); parser.add_argument("--use-mock-dino",action="store_true"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); device=torch.device(args.device); torch.set_float32_matmul_precision("high")
    model=LENSOIAModel(use_mock_dino=args.use_mock_dino,selected_layers=tuple(cfg["model"]["selected_layers"]),pretrained_weights=cfg["pretrained_weights"],factor_chunk_size=args.factor_chunk_size,evidence_tau_min=float(cfg["evidence"]["tau_min"]),evidence_tau_max=float(cfg["evidence"]["tau_max"]),evidence_topk=int(cfg["evidence"]["topk"]),region_bias_abs_max=float(cfg["evidence"]["region_bias_abs_max"]),action_logit_cap=float(cfg["training"]["action_logit_norm_cap"])).to(device).train()
    optimizer=torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],lr=1e-4)
    times=[]; samples=0
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.warmup+args.steps):
        paired=step%4==0; total=args.batch_size+(args.batch_size+1)//2 if paired else args.batch_size
        image=torch.randn(total,3,360,640,device=device); start=time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=model(image,progress=1.0); loss=out["action_logits_final"].square().mean()+out["reason_logits_formal"].square().mean()+0.01*out["state_prob"].square().mean()
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
        if device.type=="cuda": torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        if step>=args.warmup: times.append(elapsed); samples+=args.batch_size
    payload={"batch_size":args.batch_size,"factor_chunk_size":args.factor_chunk_size,"warmup":args.warmup,"steps":args.steps,"seconds_mean":sum(times)/len(times),"samples_per_second":samples/sum(times),"allocated_gb":torch.cuda.max_memory_allocated(device)/(1024**3) if device.type=="cuda" else 0.0,"reserved_gb":torch.cuda.max_memory_reserved(device)/(1024**3) if device.type=="cuda" else 0.0,"paired_view_included":True,"oom":False}
    text=json.dumps(payload); print(text)
    if args.output: Path(args.output).write_text(json.dumps(payload,indent=2),encoding="utf-8")


if __name__=="__main__": main()
