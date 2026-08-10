from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.engine.dice_common import build_dice_model, load_config
from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.engine.train_dice_oia_probe import collect, make_loader
from fate_oia.utils.aie_calibration import apply_posthoc_threshold
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.dice_artifacts import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--base-checkpoint",required=True); p.add_argument("--control-epoch-dir",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--batch-size",type=int); p.add_argument("--num-workers",type=int); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); model=build_dice_model(cfg,a.base_checkpoint,device)
    control_dir=Path(a.control_epoch_dir)
    required=("action_logits_final_test.pt","reason_logits_final_test.pt","file_names_test.json")
    missing=[name for name in required if not (control_dir/name).is_file()]
    if missing: raise FileNotFoundError(f"CONTROL reference artifacts missing before replay: {missing}")
    dataset=make_dataset(cfg,"test"); loader=make_loader(dataset,a.batch_size or int(cfg["data"]["batch_size"]),False,a.num_workers if a.num_workers is not None else int(cfg["data"]["num_workers"]),cfg)
    store={key:[] for key in ("base_action","reason","action_target","reason_target")}; current_names=[]
    model.base_model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                output=model.base_model(batch["image"].to(device,non_blocking=True),**model.base_forward_kwargs)
            for key,value in (("base_action",output["action_logits_final"]),("reason",output["reason_logits_final"]),
                              ("action_target",batch["action"]),("reason_target",batch["reason"])):
                store[key].append(value.detach().cpu())
            current_names.extend(batch["file_name"])
    current={**{key:torch.cat(value) for key,value in store.items()},"file_name":current_names}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    torch.save(current,out/"DICE_BASE_REPLAY_OUTPUTS.tmp.pt")
    reference={"action_final":torch.load(control_dir/"action_logits_final_test.pt",map_location="cpu"),
               "reason_final":torch.load(control_dir/"reason_logits_final_test.pt",map_location="cpu"),
               "file_name":json.loads((control_dir/"file_names_test.json").read_text(encoding="utf-8"))}
    branch=json.loads((Path(a.control_epoch_dir)/"branch_metrics.json").read_text(encoding="utf-8"))
    thresholds=torch.tensor(branch.get("thresholds_train_calib",branch["calibration_thresholds"]["threshold_prob"]))
    name_equal=current["file_name"]==reference["file_name"]
    action_error=float((current["base_action"]-reference["action_final"]).abs().max()); reason_error=float((current["reason"]-reference["reason_final"]).abs().max())
    metrics=aie_branch_metrics(apply_posthoc_threshold(current["base_action"],thresholds[:4]),apply_posthoc_threshold(current["reason"],thresholds[4:]),current["action_target"],current["reason_target"])
    expected=branch["deploy"]; keys=("Act_mF1","Act_oF1","Act_mAP","Exp_mF1","Exp_oF1","Exp_mAP","joint"); metric_error={key:abs(float(metrics[key])-float(expected[key])) for key in keys}
    passed=name_equal and action_error<=1e-5 and reason_error<=1e-5 and max(metric_error.values())<=1e-5
    payload={"pass":passed,"sample_count":len(dataset),"file_order_equal":name_equal,"action_logit_max_abs":action_error,"reason_logit_max_abs":reason_error,"metrics":{k:metrics[k] for k in keys},"expected":{k:expected[k] for k in keys},"metric_abs_error":metric_error}
    write_json(out/"DICE_BASE_REPLAY.json",payload); (out/"DICE_BASE_REPLAY_OUTPUTS.tmp.pt").unlink(missing_ok=True); print(json.dumps(payload,indent=2)); raise SystemExit(0 if passed else 2)


if __name__=="__main__": main()
