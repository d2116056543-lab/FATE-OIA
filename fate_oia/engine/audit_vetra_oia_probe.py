from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.engine.train_vetra_oia_probe import _optimizer
from fate_oia.engine.vetra_common import build_vetra_model, load_config, make_loader, tensor_state_hash
from fate_oia.losses.vetra_losses import total_vetra_loss
from fate_oia.losses.vetra_map_loss import VETRAMAPLoss
from fate_oia.utils.vetra_artifacts import write_json
from fate_oia.utils.vetra_contracts import assert_base_frozen, assert_vetra_contract


REQUIRED=("fate_oia/models/vetra_visual_factor_transport.py","fate_oia/models/vetra_oia_model.py",
          "fate_oia/losses/vetra_map_loss.py","fate_oia/losses/vetra_losses.py",
          "fate_oia/engine/train_vetra_oia_probe.py","fate_oia/engine/evaluate_vetra_oia_probe.py",
          "fate_oia/engine/replay_vetra_source.py")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); root=Path.cwd(); missing=[f for f in REQUIRED if not (root/f).is_file()]
    source="\n".join((root/f).read_text(encoding="utf-8") for f in REQUIRED if (root/f).is_file())
    forbidden={pattern:(pattern in source) for pattern in ("cached_logits","feature_cache_enabled: true","token_compression: keep_merge","run_c_logits","action_set_probs @")}
    model=build_vetra_model(cfg,a.source_checkpoint,device); before=tensor_state_hash(model.base_model)
    dataset=make_dataset(cfg,"train"); loader=make_loader(Subset(dataset,[0,1]),2,False,0,cfg); batch=next(iter(loader))
    images=batch["image"].to(device); target=batch["action"].to(device); map_loss=VETRAMAPLoss().to(device); opt=_optimizer(model,cfg)
    calls=[0]
    def hook(*_): calls[0]+=1
    handle=model.base_model.foundation.dino.register_forward_hook(hook); step_rows=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            out=model(images,alpha=.1); assert_vetra_contract(out)
            loss,parts=total_vetra_loss(out,target,map_loss,cfg["loss_weights"],float(cfg["vetra"]["rank_preserve_ratio"]),float(cfg["vetra"]["base_margin_floor"]),float(cfg["vetra"]["null_max_reliable"]),float(cfg["vetra"]["predicate_confidence_floor"]))
        loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.transport.parameters(),1.0)); opt.step()
        step_rows.append({"step":step,"loss":float(loss),"transport_grad_norm":grad,"finite":bool(torch.isfinite(loss))})
    handle.remove(); after=tensor_state_hash(model.base_model); assert_base_frozen(model,before,tensor_state_hash)
    checks={"required_files":not missing,"forbidden_patterns":not any(forbidden.values()),"real_dino_two_steps":calls[0]==2,
            "transport_gradient":all(row["transport_grad_norm"]>0 for row in step_rows),"finite":all(row["finite"] for row in step_rows),
            "base_hash_unchanged":before==after,"base_grad_zero":all(p.grad is None for p in model.base_model.parameters()),
            "reason_exact":torch.equal(out["reason_logits_base"],out["reason_logits_final"]),"correction_cap":float(out["vetra_action_delta"].abs().max())<=.200001}
    payload={"pass":all(checks.values()),"checks":checks,"missing":missing,"forbidden":forbidden,"steps":step_rows,"dino_calls":calls[0]}
    write_json(Path(a.output_dir)/"VETRA_IMPLEMENTATION_AUDIT.json",payload); print(json.dumps(payload,indent=2))
    if not payload["pass"]: raise SystemExit(1)


if __name__ == "__main__": main()
