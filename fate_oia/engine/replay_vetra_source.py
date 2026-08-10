from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.engine.vetra_common import build_vetra_model, load_config, make_loader
from fate_oia.utils.vetra_artifacts import write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--max-samples",type=int); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); model=build_vetra_model(cfg,a.source_checkpoint,device).eval()
    source=Path(cfg["experiment"]["source_control_dir"]); dataset=make_dataset(cfg,"test")
    if a.max_samples: dataset=Subset(dataset,list(range(min(a.max_samples,len(dataset)))))
    loader=make_loader(dataset,6,False,int(cfg["data"]["num_workers"]),cfg)
    action=[]; reason=[]; names=[]
    for batch in loader:
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=model.base_model(batch["image"].to(device,non_blocking=True),**model.base_forward_kwargs)
        action.append(out["action_logits_final"].cpu()); reason.append(out["reason_logits_final"].cpu()); names.extend(batch["file_name"])
    action=torch.cat(action); reason=torch.cat(reason)
    ref_action=torch.load(source/"action_logits_final_test.pt",weights_only=True)[:len(names)]; ref_reason=torch.load(source/"reason_logits_final_test.pt",weights_only=True)[:len(names)]
    import json
    ref_names=json.loads((source/"file_names_test.json").read_text(encoding="utf-8"))[:len(names)]
    payload={"pass":bool(torch.equal(action,ref_action) and torch.equal(reason,ref_reason) and names==ref_names),
             "batch_size":6,"samples":len(names),"full_test":len(names)==4572,"action_exact":torch.equal(action,ref_action),
             "reason_exact":torch.equal(reason,ref_reason),"file_order_exact":names==ref_names,
             "action_max_abs":float((action-ref_action).abs().max()),"reason_max_abs":float((reason-ref_reason).abs().max())}
    write_json(Path(a.output_dir)/"VETRA_SOURCE_REPLAY.json",payload); print(payload)
    if not payload["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
