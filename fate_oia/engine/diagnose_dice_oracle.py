from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.engine.dice_common import build_dice_model, load_config
from fate_oia.engine.train_aie_oia import collate, make_dataset
from fate_oia.engine.train_dice_oia_probe import make_loader
from fate_oia.losses.dice_losses import certificate_targets
from fate_oia.utils.aie_counterfactual import AIECounterfactualConfig, AIECounterfactualEngine
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.dice_artifacts import write_json


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--base-checkpoint",required=True); p.add_argument("--max-audit-samples",type=int,default=1024); p.add_argument("--output-dir",required=True); p.add_argument("--device",default="cuda"); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); wrapper=build_dice_model(cfg,a.base_checkpoint,device); base=wrapper.base_model.eval()
    dataset=make_dataset(cfg,"train"); names=[s.file_name for s in dataset.samples]; split=stable_split_ids(names,int(cfg["data"]["seed"]),float(cfg["data"]["train_calib_fraction"]),int(cfg["data"]["train_audit_count"])); index={s.file_name:i for i,s in enumerate(dataset.samples)}
    ids=[index[n] for n in split["train_audit"][:a.max_audit_samples]]; loader=make_loader(Subset(dataset,ids),int(cfg["data"]["batch_size"]),False,int(cfg["data"]["num_workers"]),cfg)
    engine=AIECounterfactualEngine(AIECounterfactualConfig(batch_fraction=1,max_actions_per_sample=2,max_atoms_per_event=2))
    base_logits=[]; oracle_logits=[]; effect_oracle_logits=[]; benefit_oracle_logits=[]; actions=[]; reasons=[]; reason_logits=[]; valid=0; high=0; per_action={str(i):[] for i in range(4)}
    direction_counts={name:{"kept":0,"suppressed":0,"matched":0,"mismatched":0} for name in ("support","counter")}
    with torch.no_grad():
        for batch in loader:
            image=batch["image"].to(device,non_blocking=True); action=batch["action"].to(device)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                output=base(image,action_scale=1.,reason_scale=.60)
                support_first=engine.run(base,output,action,batch["file_name"],global_update=17,action_scale=1.)
                support_second=engine.run(base,output,action,batch["file_name"],global_update=41,action_scale=1.)
                # Negating only the selector tensor makes the unchanged AIE engine choose
                # the strongest target-counter atom without changing any rerun logits.
                counter_output={**output,"bounded_contribution":-output["bounded_contribution"]}
                counter_first=engine.run(base,counter_output,action,batch["file_name"],global_update=17,action_scale=1.)
                counter_second=engine.run(base,counter_output,action,batch["file_name"],global_update=41,action_scale=1.)
            adjusted=output["action_logits_final"].clone()
            effect_sum=torch.zeros_like(adjusted)
            benefit_sum=torch.zeros_like(adjusted)
            sample_lookup={name:i for i,name in enumerate(batch["file_name"])}
            for direction,first,second,selector_sign in (("support",support_first,support_second,1.0),("counter",counter_first,counter_second,-1.0)):
                second_rows={(r["file_name"],r["action_id"],r["probe_id"]):r for r in second["cases"]}
                for row in first["cases"]:
                    key=(row["file_name"],row["action_id"],row["probe_id"]); other=second_rows.get(key,row)
                    controls=torch.tensor([row["control_drop"],other["control_drop"],row["wrong_probe_drop"],row["wrong_action_drop"]],device=device)
                    selected=torch.tensor(row["selected_drop"],device=device); cert=certificate_targets(selected[None],controls[None],float(cfg["dice"]["license_temperature"]))
                    effect=float((selected-cert["control_median"][0]).cpu()); action_id=int(row["action_id"]); sample_id=sample_lookup[row["file_name"]]
                    target_sign=2*action[sample_id,action_id]-1
                    signed_legacy=selector_sign*float(row["supportive_contribution"])
                    contribution=target_sign*torch.tensor(signed_legacy,device=device)
                    matched=signed_legacy*effect>0
                    confidence=max(float(cert["license_support_cf"][0]),float(cert["license_counter_cf"][0]))
                    keep=1.0 if matched and confidence>=.5 else 0.0
                    adjusted[sample_id,action_id]-=contribution*(1-keep)
                    effect_sum[sample_id,action_id]+=target_sign*torch.tensor(effect,device=device).clamp(-.08,.08)
                    benefit_sum[sample_id,action_id]+=target_sign*torch.tensor(max(effect,0.0),device=device).clamp_max(.08)
                    direction_counts[direction]["matched" if matched else "mismatched"]+=1
                    direction_counts[direction]["kept" if keep else "suppressed"]+=1
                    valid+=1; high+=int(confidence>=.7); per_action[str(action_id)].append(effect)
            effect_adjusted=output["action_logits_final"]+.25*torch.tanh(effect_sum/.25)
            benefit_adjusted=output["action_logits_final"]+.25*torch.tanh(benefit_sum/.25)
            base_logits.append(output["action_logits_final"].cpu()); oracle_logits.append(adjusted.cpu()); effect_oracle_logits.append(effect_adjusted.cpu()); benefit_oracle_logits.append(benefit_adjusted.cpu()); actions.append(action.cpu()); reason_logits.append(output["reason_logits_final"].cpu()); reasons.append(batch["reason"])
    base_action=torch.cat(base_logits); oracle_action=torch.cat(oracle_logits); effect_oracle_action=torch.cat(effect_oracle_logits); benefit_oracle_action=torch.cat(benefit_oracle_logits); action=torch.cat(actions); reason_logit=torch.cat(reason_logits); reason=torch.cat(reasons)
    base_metric=aie_branch_metrics(base_action,reason_logit,action,reason); oracle_metric=aie_branch_metrics(oracle_action,reason_logit,action,reason); effect_oracle_metric=aie_branch_metrics(effect_oracle_action,reason_logit,action,reason); benefit_oracle_metric=aie_branch_metrics(benefit_oracle_action,reason_logit,action,reason)
    directions={key:(sum(values)/len(values) if values else None) for key,values in per_action.items()}
    passed=(oracle_metric["Act_mF1"]>=base_metric["Act_mF1"]+.004 and oracle_metric["Act_mAP"]>=base_metric["Act_mAP"]+.002 and all(v is not None and v>0 for v in directions.values()) and high/max(valid,1)>=.30)
    effect_pass=(effect_oracle_metric["Act_mF1"]>=base_metric["Act_mF1"]+.004 and effect_oracle_metric["Act_mAP"]>=base_metric["Act_mAP"]+.002)
    benefit_pass=(benefit_oracle_metric["Act_mF1"]>=base_metric["Act_mF1"]+.004 and benefit_oracle_metric["Act_mAP"]>=base_metric["Act_mAP"]+.002)
    result={"pass":passed,"legacy_selection_pass":passed,"dice_effect_oracle_pass":effect_pass,
            "dice_benefit_oracle_pass":benefit_pass,"base":base_metric,"oracle":oracle_metric,"dice_effect_oracle":effect_oracle_metric,"dice_benefit_oracle":benefit_oracle_metric,
            "delta_Act_mF1":oracle_metric["Act_mF1"]-base_metric["Act_mF1"],"delta_Act_mAP":oracle_metric["Act_mAP"]-base_metric["Act_mAP"],
            "effect_oracle_delta_Act_mF1":effect_oracle_metric["Act_mF1"]-base_metric["Act_mF1"],
            "effect_oracle_delta_Act_mAP":effect_oracle_metric["Act_mAP"]-base_metric["Act_mAP"],
            "benefit_oracle_delta_Act_mF1":benefit_oracle_metric["Act_mF1"]-base_metric["Act_mF1"],
            "benefit_oracle_delta_Act_mAP":benefit_oracle_metric["Act_mAP"]-base_metric["Act_mAP"],
            "per_action_direction":directions,"valid_events":valid,"high_confidence_rate":high/max(valid,1),"direction_counts":direction_counts,"test_used":False}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); write_json(out/"DICE_ORACLE_POTENTIAL.json",result); print(json.dumps(result,indent=2)); raise SystemExit(0 if passed else 2)


if __name__=="__main__": main()
