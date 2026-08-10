from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.engine.dice_common import build_dice_model, load_config, tensor_state_hash
from fate_oia.engine.train_aie_oia import collate, make_dataset
from fate_oia.losses.dice_loss_registry import DICELossRegistry
from fate_oia.losses.dice_losses import action_asl_loss, delta_regularizer, directional_effect_loss, directional_license_loss, route_directional_license_targets
from fate_oia.losses.dice_rank_sketch import DistributionalRankSketch, quantile_rank_preservation_loss
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds
from fate_oia.utils.aie_metrics import aie_branch_metrics, spearman_correlation
from fate_oia.utils.dice_artifacts import append_jsonl, sha256, write_json
from fate_oia.utils.dice_contracts import assert_base_frozen, assert_probe_contract
from fate_oia.utils.dice_counterfactual_engine import DICECounterfactualEngine
from fate_oia.utils.dice_metrics import mechanism_metrics


def binary_auc(score: torch.Tensor, target: torch.Tensor) -> float | None:
    positive, negative = target > .5, target <= .5
    if not bool(positive.any() and negative.any()): return None
    return float(((score[positive,None] > score[negative][None]).float() + .5*(score[positive,None] == score[negative][None]).float()).mean())


def summarize_cf_rows(rows: list[dict]) -> dict:
    if not rows:
        return {"valid_events":0,"per_action":{str(a):{"count":0,"effect_mean":None} for a in range(4)},
                "certificate_mean":None,"certificate_positive_rate":None,"certificate_positive_rate_lcb95":None,
                "license_prediction_auc":None,"contribution_effect_spearman":None}
    effects=torch.tensor([row["effect"] for row in rows]); positive=(effects>0).float()
    rate=float(positive.mean()); lcb=rate-1.96*math.sqrt(max(rate*(1-rate)/len(rows),0))
    per_action={str(action):{"count":sum(row["action_id"]==action for row in rows),
        "effect_mean":float(np.mean([row["effect"] for row in rows if row["action_id"]==action]))
        if any(row["action_id"]==action for row in rows) else None} for action in range(4)}
    aucs=[]
    for prediction,target in (("support_hat","support_target"),("counter_hat","counter_target")):
        value=binary_auc(torch.tensor([row[prediction] for row in rows]),torch.tensor([row[target] for row in rows]))
        if value is not None: aucs.append(value)
    return {"valid_events":len(rows),"per_action":per_action,"certificate_mean":float(effects.mean()),
            "certificate_positive_rate":rate,"certificate_positive_rate_lcb95":lcb,
            "license_prediction_auc":float(np.mean(aucs)) if aucs else None,
            "contribution_effect_spearman":spearman_correlation(
                torch.tensor([row["target_signed_contribution"] for row in rows]),effects)}


def restore_training_state(model, optimizer, sketch, checkpoint: str | Path, scheduler=None, base_sketch=None) -> tuple[int,int]:
    state=torch.load(checkpoint,map_location=next(model.parameters()).device,weights_only=False)
    model.load_state_dict(state["model"],strict=True); optimizer.load_state_dict(state["optimizer"])
    sketch.load_state_dict(state["rank_sketch"])
    if base_sketch is not None and state.get("base_rank_sketch") is not None:
        base_sketch.load_state_dict(state["base_rank_sketch"])
    if scheduler is not None and state.get("scheduler") is not None: scheduler.load_state_dict(state["scheduler"])
    return int(state["epoch"])+1,int(state["update"])


def make_scheduler(optimizer, total_updates: int, warmup_ratio: float, min_lr_ratio: float):
    warmup=max(1,int(total_updates*warmup_ratio))
    def factor(step):
        if step<warmup: return max((step+1)/warmup,1e-8)
        progress=min(max((step-warmup)/max(total_updates-warmup,1),0.0),1.0)
        return min_lr_ratio+(1-min_lr_ratio)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def macro_pair_inversion(scores: torch.Tensor, target: torch.Tensor) -> float | None:
    rates=[]
    for label in range(target.shape[1]):
        pos,neg=target[:,label]>.5,target[:,label]<=.5
        if bool(pos.any() and neg.any()): rates.append((scores[pos,label,None]<=scores[neg,label][None]).float().mean())
    return float(torch.stack(rates).mean()) if rates else None


def gradient_norm(parameters) -> float:
    values=[parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(values).norm()) if values else 0.0


def frozen_parameter_delta_max(module, versions: dict[str,int]) -> float:
    changed=[name for name,value in module.named_parameters() if value._version!=versions[name]]
    if changed: raise RuntimeError(f"frozen base parameters were mutated: {changed[:8]}")
    return 0.0


def make_loader(dataset, batch: int, shuffle: bool, workers: int, cfg: dict, generator=None):
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                  pin_memory=bool(cfg["data"]["pin_memory"]), generator=generator,
                  persistent_workers=bool(cfg["data"]["persistent_workers"]) and workers > 0)
    if workers:
        kwargs["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


@torch.no_grad()
def collect(model, loader, device):
    model.eval(); store = {key: [] for key in ("base_action","dice_action","reason","action_target","reason_target")}; names=[]
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(batch["image"].to(device, non_blocking=True))
        assert_probe_contract(out)
        for key, value in (("base_action",out["action_logits_base"]),("dice_action",out["action_logits_final"]),
                           ("reason",out["reason_logits_final"]),("action_target",batch["action"]),("reason_target",batch["reason"])):
            store[key].append(value.detach().cpu())
        names.extend(batch["file_name"])
    return {**{k:torch.cat(v) for k,v in store.items()},"file_name":names}


@torch.no_grad()
def evaluate(model, calib_loader, test_loader, device, epoch_dir: Path):
    calib, test = collect(model, calib_loader, device), collect(model, test_loader, device)
    base_threshold = fit_posthoc_thresholds(torch.cat((calib["base_action"],calib["reason"]),1),
        torch.cat((calib["action_target"],calib["reason_target"]),1), [list(range(4)),list(range(4,25))])["threshold_prob"]
    dice_threshold = fit_posthoc_thresholds(torch.cat((calib["dice_action"],calib["reason"]),1),
        torch.cat((calib["action_target"],calib["reason_target"]),1), [list(range(4)),list(range(4,25))])["threshold_prob"]
    base_raw = aie_branch_metrics(test["base_action"],test["reason"],test["action_target"],test["reason_target"])
    dice_raw = aie_branch_metrics(test["dice_action"],test["reason"],test["action_target"],test["reason_target"])
    base_deploy = aie_branch_metrics(apply_posthoc_threshold(test["base_action"],base_threshold[:4]),
        apply_posthoc_threshold(test["reason"],base_threshold[4:]),test["action_target"],test["reason_target"])
    dice_deploy = aie_branch_metrics(apply_posthoc_threshold(test["dice_action"],dice_threshold[:4]),
        apply_posthoc_threshold(test["reason"],dice_threshold[4:]),test["action_target"],test["reason_target"])
    payload={"base_raw":base_raw,"dice_raw":dice_raw,"base_deploy":base_deploy,"dice_deploy":dice_deploy,
             "base_thresholds_train_calib":base_threshold.tolist(),"dice_thresholds_train_calib":dice_threshold.tolist(),
             "reason_identity_max_abs":0.0}
    epoch_dir.mkdir(parents=True,exist_ok=True)
    for name,key in (("action_logits_base_test.pt","base_action"),("action_logits_dice_test.pt","dice_action"),
                     ("reason_logits_base_test.pt","reason"),("reason_logits_dice_test.pt","reason"),
                     ("labels_action_test.pt","action_target"),("labels_reason_test.pt","reason_target")):
        torch.save(test[key],epoch_dir/name)
    (epoch_dir/"file_names_test.json").write_text(json.dumps(test["file_name"],ensure_ascii=False),encoding="utf-8")
    write_json(epoch_dir/"branch_metrics.json",payload)
    return payload


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--base-checkpoint",required=True)
    parser.add_argument("--output-dir",required=True); parser.add_argument("--epochs",type=int); parser.add_argument("--batch-size",type=int)
    parser.add_argument("--gradient-accumulation-steps",type=int); parser.add_argument("--num-workers",type=int)
    parser.add_argument("--max-train-samples",type=int); parser.add_argument("--max-calib-samples",type=int); parser.add_argument("--max-test-samples",type=int)
    parser.add_argument("--device",default="cuda"); parser.add_argument("--resume")
    args=parser.parse_args(); cfg=load_config(args.config); out_dir=Path(args.output_dir); out_dir.mkdir(parents=True,exist_ok=True)
    seed=int(cfg["data"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=torch.device(args.device); torch.set_num_threads(int(cfg["runtime"]["cpu_threads"])); torch.set_float32_matmul_precision("high")
    if cfg["experiment"]["feature_cache_enabled"] or cfg["experiment"]["token_compression"]!="none": raise RuntimeError("DICE forbids cache/compression")
    model=build_dice_model(cfg,args.base_checkpoint,device); assert_base_frozen(model); base_hash=tensor_state_hash(model.base_model)
    base_versions={name:value._version for name,value in model.base_model.named_parameters()}
    optimizer=torch.optim.AdamW([
        {"params":model.atom_reconstructor.parameters(),"lr":float(cfg["training"]["lr_atom_reconstructor"])},
        {"params":[model.directional_head.support_weight,model.directional_head.counter_weight],"lr":float(cfg["training"]["lr_directional_head"])},
        {"params":model.directional_head.license.parameters(),"lr":float(cfg["training"]["lr_license_predictor"])}],weight_decay=float(cfg["training"]["weight_decay"]))
    sketch=DistributionalRankSketch(4,int(cfg["dice"]["rank_quantiles"])).to(device)
    base_sketch=DistributionalRankSketch(4,int(cfg["dice"]["rank_quantiles"])).to(device)
    cf_engine=DICECounterfactualEngine(float(cfg["dice"]["license_temperature"]),int(cfg["counterfactual"]["max_actions_per_sample"]),
        float(cfg["counterfactual"]["batch_fraction"]),int(cfg["counterfactual"].get("topk_patches",64)))
    full_train,full_test=make_dataset(cfg,"train"),make_dataset(cfg,"test"); names=[s.file_name for s in full_train.samples]
    split=stable_split_ids(names,seed,float(cfg["data"]["train_calib_fraction"]),int(cfg["data"]["train_audit_count"])); index={s.file_name:i for i,s in enumerate(full_train.samples)}
    train_ids=[index[n] for n in split["train_main"][:args.max_train_samples or None]]
    calib_ids=[index[n] for n in split["train_calib"][:args.max_calib_samples or None]]; test_ids=list(range(len(full_test)))[:args.max_test_samples or None]
    batch=args.batch_size or int(cfg["data"]["batch_size"]); workers=args.num_workers if args.num_workers is not None else int(cfg["data"]["num_workers"])
    gen=torch.Generator().manual_seed(seed); train_loader=make_loader(Subset(full_train,train_ids),batch,True,workers,cfg,gen)
    calib_loader=make_loader(Subset(full_train,calib_ids),batch,False,workers,cfg); test_loader=make_loader(Subset(full_test,test_ids),batch,False,workers,cfg)
    split_path=out_dir/"split_manifest.json"; write_split_manifest(split_path,names,seed,float(cfg["data"]["train_calib_fraction"]),int(cfg["data"]["train_audit_count"]))
    accumulation=args.gradient_accumulation_steps or int(cfg["data"]["gradient_accumulation_steps"]); epochs=args.epochs or int(cfg["training"]["epochs"])
    manifest={"git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"source_head":cfg["experiment"]["source_head"],
              "base_checkpoint":str(Path(args.base_checkpoint).resolve()),"base_checkpoint_hash":sha256(args.base_checkpoint),"base_hash":base_hash,
              "config_hash":sha256(args.config),"split_hash":sha256(split_path),"train_samples":len(train_ids),"calib_samples":len(calib_ids),
              "audit_samples":len(split["train_audit"]),"train_calib_overlap":0,
              "train_audit_overlap":0,"test_samples":len(test_ids),
              "batch_size":batch,"gradient_accumulation_steps":accumulation,"num_workers":workers,"feature_cache_enabled":False,"token_compression":"none","best_selection_split":"test"}
    write_json(out_dir/"run_manifest.json",manifest)
    updates_per_epoch=math.ceil(len(train_loader)/accumulation); scheduler=make_scheduler(
        optimizer,updates_per_epoch*epochs,float(cfg["training"]["warmup_ratio"]),float(cfg["training"]["min_lr_ratio"]))
    start_epoch=0; update=0
    if args.resume: start_epoch,update=restore_training_state(model,optimizer,sketch,args.resume,scheduler,base_sketch)
    all_cf_rows=[]
    for previous in sorted(out_dir.glob("epoch_*/dice_counterfactual_cases.json")):
        all_cf_rows.extend(json.loads(previous.read_text(encoding="utf-8")))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,epochs):
        model.train(); start=time.perf_counter(); mechanism=[]; cf_rows=[]; batch_end=time.perf_counter(); pending_rank=[]
        for micro,batch_data in enumerate(train_loader):
            data_seconds=time.perf_counter()-batch_end
            will_step=(micro+1)%accumulation==0 or micro+1==len(train_loader)
            profile_due=will_step and (update+1==1 or (update+1)%int(cfg["training"]["print_every_optimizer_updates"])==0)
            action=batch_data["action"].to(device,non_blocking=True); images=batch_data["image"].to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                if profile_due and device.type=="cuda": torch.cuda.synchronize()
                base_start=time.perf_counter()
                with torch.no_grad(): base_output=model.base_model(images,**model.base_forward_kwargs)
                if profile_due and device.type=="cuda": torch.cuda.synchronize()
                base_seconds=time.perf_counter()-base_start; dice_start=time.perf_counter(); output=model.decode_base_output(base_output)
                if profile_due and device.type=="cuda": torch.cuda.synchronize()
                dice_seconds=time.perf_counter()-dice_start; cf_due=(micro+1)%accumulation==0 and (update+1)%int(cfg["counterfactual"]["interval_optimizer_updates"])==0
                cf_start=time.perf_counter(); cf=cf_engine.run(model,output,action,update) if cf_due else None
                if profile_due and device.type=="cuda": torch.cuda.synchronize()
                cf_seconds=time.perf_counter()-cf_start
                registry=DICELossRegistry(); registry.add("action_asl",action_asl_loss(output["action_logits_final"],action))
                registry.add("rank_sketch",sketch.loss(output["action_logits_final"],action,float(cfg["dice"]["rank_margin"])))
                registry.add("rank_protect",quantile_rank_preservation_loss(output["action_logits_final"],output["action_logits_base"],action,
                    sketch,base_sketch,float(cfg["dice"]["rank_preserve_ratio"]),float(cfg["dice"]["rank_margin"])))
                if cf and cf["available"]:
                    ids=cf["atom_index"]; license_logits=torch.stack([output["license_logits"][s,a,p] for s,a,p in ids]); support_logits,counter_logits=license_logits.unbind(-1)
                    support,counter=support_logits.sigmoid(),counter_logits.sigmoid(); atoms=torch.stack([output["atom_correction"][s,a,p] for s,a,p in ids]); targets=torch.stack([action[s,a] for s,a,p in ids])[:,None]
                    raw_support_target,raw_counter_target=route_directional_license_targets(
                        cf["license_support_cf"],cf["license_counter_cf"],targets[:,0])
                    registry.add("license",directional_license_loss(support_logits,counter_logits,cf["license_support_cf"],cf["license_counter_cf"],targets[:,0]))
                    registry.add("effect",directional_effect_loss(atoms,targets[:,0],cf["directional_effect"]))
                    for row,(sample_id,action_id,probe_id) in enumerate(ids):
                        cf_rows.append({"action_id":action_id,"probe_id":probe_id,"selected_drop":float(cf["selected_drop"][row].detach()),
                                        "control_median":float(cf["control_median"][row].detach()),"control_mad":float(cf["control_mad"][row].detach()),
                                        "effect":float(cf["directional_effect"][row].detach()),"support_target":float(raw_support_target[row]),
                                        "counter_target":float(raw_counter_target[row]),"target_relative_support":float(cf["license_support_cf"][row].detach()),
                                        "target_relative_counter":float(cf["license_counter_cf"][row].detach()),"support_hat":float(support[row].detach()),
                                        "counter_hat":float(counter[row].detach()),"target_signed_contribution":float(((2*action[sample_id,action_id]-1)*atoms[row]).detach())})
                else:
                    zero=output["dice_action_delta"].sum()*0; registry.add("license",zero); registry.add("effect",zero)
                registry.add("delta",delta_regularizer(output["dice_action_delta"])); loss=registry.total()/accumulation
            if profile_due and device.type=="cuda": torch.cuda.synchronize()
            backward_start=time.perf_counter(); loss.backward()
            if profile_due and device.type=="cuda": torch.cuda.synchronize()
            backward_seconds=time.perf_counter()-backward_start
            pending_rank.append((output["action_logits_final"].detach(),output["action_logits_base"].detach(),action.detach()))
            mechanism.append(mechanism_metrics(output,cf["directional_effect"] if cf and cf["available"] else None))
            if (micro+1)%accumulation==0 or micro+1==len(train_loader):
                trainable=[p for p in model.parameters() if p.requires_grad]
                nonfinite=[name for name,parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
                if nonfinite: raise FloatingPointError(f"non-finite DICE gradients before optimizer step: {nonfinite}")
                dice_grad=gradient_norm(trainable)
                dino_grad=gradient_norm(model.base_model.parameters())
                torch.nn.utils.clip_grad_norm_(trainable,float(cfg["training"]["global_grad_clip"])); optimizer.step(); optimizer.zero_grad(set_to_none=True); update+=1
                scheduler.step(); rank_final=torch.cat([row[0] for row in pending_rank]); rank_base=torch.cat([row[1] for row in pending_rank]); rank_target=torch.cat([row[2] for row in pending_rank])
                sketch.update(rank_final,rank_target,update); base_sketch.update(rank_base,rank_target,update); pending_rank.clear()
                if update==1 or update%int(cfg["training"]["print_every_optimizer_updates"])==0:
                    loss_values={f"loss_{name}":float(value.detach()) for name,value in registry.terms.items()}
                    cf_values={"cf_available":bool(cf and cf["available"])}
                    if cf and cf["available"]:
                        license_prediction=torch.cat((support.detach(),counter.detach())); license_target=torch.cat((raw_support_target,raw_counter_target))
                        license_auc=binary_auc(license_prediction,license_target)
                        per_action_certificate={str(action_id):float(torch.stack([cf["directional_effect"][row] for row,(_,event_action,_) in enumerate(ids) if event_action==action_id]).mean())
                            if any(event_action==action_id for _,event_action,_ in ids) else None for action_id in range(4)}
                        cf_values.update({"selected_drop_mean":float(cf["selected_drop"].detach().mean()),
                            "control_median_mean":float(cf["control_median"].detach().mean()),
                            "control_mad_mean":float(cf["control_mad"].detach().mean()),
                            "directional_effect_mean":float(cf["directional_effect"].detach().mean()),
                            "directional_certificate":float(cf["directional_effect"].detach().mean()),
                            "certificate_positive_rate":float((cf["directional_effect"].detach()>0).float().mean()),
                            "per_action_certificate":per_action_certificate,"license_prediction_auc":license_auc,
                            "license_prediction_spearman":spearman_correlation(license_prediction.float(),license_target.float()),
                            "license_support_cf_mean":float(cf["license_support_cf"].detach().mean()),
                            "license_counter_cf_mean":float(cf["license_counter_cf"].detach().mean())})
                    delta=output["dice_action_delta"].detach().float(); quantile=torch.quantile(delta.flatten(),torch.tensor([.1,.5,.9],device=delta.device))
                    diagnostic={"base_hash":base_hash,"base_action_logit_rms":tensor_rms(output["action_logits_base"]),
                        "dice_action_logit_rms":tensor_rms(output["action_logits_final"]),"dice_delta_rms":tensor_rms(delta),
                        "dice_delta_p10":float(quantile[0]),"dice_delta_p50":float(quantile[1]),"dice_delta_p90":float(quantile[2]),
                        "per_action_delta_mean":delta.mean(0).tolist(),"coherent_token_norm":float(output["coherent_token"].detach().float().norm(dim=-1).mean()),
                        "background_centering_norm":float(output["centered_token"].detach().float().norm(dim=-1).mean()),
                        "base_pair_inversion_rate":macro_pair_inversion(output["action_logits_base"].detach(),action),
                        "dice_pair_inversion_rate":macro_pair_inversion(output["action_logits_final"].detach(),action),
                        "base_parameter_delta_max":frozen_parameter_delta_max(model.base_model,base_versions),"dice_grad_norm":dice_grad,"dino_grad":dino_grad,
                        "allocated_gb":torch.cuda.memory_allocated()/2**30 if device.type=="cuda" else 0.0,
                        "reserved_gb":torch.cuda.memory_reserved()/2**30 if device.type=="cuda" else 0.0,
                        "data_time":data_seconds,"dino_time":base_seconds,"dice_time":dice_seconds,"cf_time":cf_seconds,"backward_time":backward_seconds}
                    row={"dice_batch":True,"epoch":epoch,"update":update,"lr":optimizer.param_groups[0]["lr"],
                         "loss_total":float(loss.detach()*accumulation),**loss_values,**cf_values,
                         **diagnostic,**mechanism[-1],**sketch.stats(update)}
                    row.update({"action_asl":loss_values["loss_action_asl"],"rank_repair_loss":loss_values["loss_rank_sketch"],
                        "rank_protect_loss":loss_values["loss_rank_protect"],"license_loss":loss_values["loss_license"],
                        "effect_loss":loss_values["loss_effect"],"delta_loss":loss_values["loss_delta"],
                        "rank_sketch_positive_count":row["positive_count"],"rank_sketch_negative_count":row["negative_count"],
                        "rank_sketch_age":row["age"]})
                    print(json.dumps(row),flush=True); append_jsonl(out_dir/"loss_components.jsonl",row)
            batch_end=time.perf_counter()
        if tensor_state_hash(model.base_model)!=base_hash: raise RuntimeError("frozen base hash changed")
        epoch_dir=out_dir/f"epoch_{epoch:03d}"; metrics=evaluate(model,calib_loader,test_loader,device,epoch_dir)
        all_cf_rows.extend(cf_rows); epoch_mech=summarize_cf_rows(cf_rows); mech=summarize_cf_rows(all_cf_rows)
        write_json(epoch_dir/"dice_counterfactual_cases.json",cf_rows)
        write_json(epoch_dir/"dice_mechanism_metrics.json",{"updates":len(mechanism),"last":mechanism[-1],"rank_sketch":sketch.stats(update),"base_hash_unchanged":True,"epoch_mechanism":epoch_mech,"across_run":mech})
        write_json(out_dir/"DICE_MECHANISM_GATES.json",mech)
        append_jsonl(out_dir/"metrics_summary.jsonl",{"epoch":epoch,**metrics}); print(json.dumps({"dice_epoch":epoch,**metrics}),flush=True)
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
               "rank_sketch":sketch.state_dict(),"base_rank_sketch":base_sketch.state_dict(),"epoch":epoch,"update":update,"manifest":manifest}
        torch.save(state,out_dir/"checkpoint_latest.pth"); torch.save(state,out_dir/f"checkpoint_epoch_{epoch:03d}.pth")


if __name__=="__main__": main()
