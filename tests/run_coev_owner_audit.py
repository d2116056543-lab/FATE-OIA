from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch

from fate_oia.datasets.coev_video_dataset import CoEVVideoDataset, coev_collate
from fate_oia.engine.train_coev_oia import build_model, load_config
from fate_oia.losses.coev_losses import CoEVLoss
from fate_oia.utils.coev_preflight import source_identity


def norm(parameters, grads) -> float:
    device = next((g.device for g in grads if g is not None), parameters[0].device)
    total = torch.zeros((), device=device)
    for grad in grads:
        if grad is not None: total += grad.float().square().sum()
    return float(total.sqrt())


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/coev_oia_v1.yaml");parser.add_argument("--output",required=True);args=parser.parse_args()
    cfg=load_config(args.config)
    ds=CoEVVideoDataset(cfg["data"]["manifest_path"],["train_core","train_calib","train_audit"],True,cfg["data"]["grounding_root"],max_samples=1)
    inputs,targets=coev_collate([ds[0]]); inputs=inputs.to("cuda"); targets=targets.to("cuda")
    model=build_model(cfg).cuda().train();model.capture_gradient_diagnostics=True
    hook_counts={"block4":0,"block8":0,"block12":0};handles=[]
    for index,name in ((3,"block4"),(7,"block8"),(11,"block12")):
        def hook(_module,_inputs,_output,key=name): hook_counts.__setitem__(key,hook_counts[key]+1)
        handles.append(model.visual_field.backbone.blocks[index].register_forward_hook(hook))
    with torch.autocast("cuda",dtype=torch.bfloat16): output=model(inputs)
    for handle in handles:handle.remove()
    losses=CoEVLoss()(output,targets)
    owners={
      "dino9":list(model.visual_field.backbone.blocks[8].parameters()),
      "dino10":list(model.visual_field.backbone.blocks[9].parameters()),
      "dino11":list(model.visual_field.backbone.blocks[10].parameters()),
      "dino12":list(model.visual_field.backbone.blocks[11].parameters()),
      "observer":list(model.predicate_observer.parameters()),
      "matcher":list(model.correspondence_observer.parameters()),
      "decoder":list(model.video_decoder.parameters()),
      "action_readout":[model.evidence_readout.coeff_action],
      "reason_readout":[model.evidence_readout.coeff_reason],
    }
    result={}
    history_grads=torch.autograd.grad(losses["action_task"],output["history_gradient_fields"],retain_graph=True,allow_unused=True)
    result["selected_block_hook_counts"]=hook_counts
    result["history_layer_frame_grad"]=[{"early":float(g[:,0].float().norm()),"middle":float(g[:,6].float().norm()),"late":float(g[:,13].float().norm())} for g in history_grads]
    for loss_name in ("action_task","reason_task","ground","match"):
        params=[p for values in owners.values() for p in values]
        grads=torch.autograd.grad(losses[loss_name],params,retain_graph=True,allow_unused=True)
        row={}; offset=0
        for owner,values in owners.items(): row[owner]=norm(values,grads[offset:offset+len(values)]); offset+=len(values)
        result[loss_name]=row
    result["pass"]=(all(result["action_task"][f"dino{i}"]>0 for i in range(9,13)) and all(v>0 for v in hook_counts.values()) and
                     all(all(v>0 for v in row.values()) for row in result["history_layer_frame_grad"]) and
                     result["action_task"]["observer"]==0 and result["action_task"]["matcher"]==0 and
                     result["reason_task"]["action_readout"]==0 and result["action_task"]["reason_readout"]==0 and
                     result["ground"]["observer"]>0 and result["match"]["matcher"]>0)
    result["audit_identity"]=source_identity(args.config)
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp");temp.write_text(json.dumps(result,indent=2),encoding="utf-8");temp.replace(path)
    print(json.dumps(result,indent=2))
    if not result["pass"]: raise SystemExit(2)


if __name__=="__main__":main()
