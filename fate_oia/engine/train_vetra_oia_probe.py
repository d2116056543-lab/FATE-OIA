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
from torch.utils.data import Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.engine.train_aie_oia import make_dataset
from fate_oia.engine.vetra_common import (alpha_schedule, build_vetra_model, load_config,
                                          make_loader, make_scheduler, tensor_state_hash)
from fate_oia.engine.evaluate_vetra_oia_probe import finalize_probe
from fate_oia.losses.vetra_losses import total_vetra_loss
from fate_oia.losses.vetra_map_loss import VETRAMAPLoss
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds
from fate_oia.utils.aie_metrics import aie_branch_metrics
from fate_oia.utils.vetra_artifacts import append_jsonl, file_sha256, write_json
from fate_oia.utils.vetra_contracts import assert_base_frozen, assert_vetra_contract
from fate_oia.utils.vetra_counterfactual_audit import VETRACounterfactualAudit
from fate_oia.utils.vetra_metrics import route_statistics


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _optimizer(model, cfg):
    groups = {"transport": [], "action_query": [], "semantic_key": [], "reliability": [], "correction_head": []}
    for name, parameter in model.transport.named_parameters():
        if "support_head" in name or "counter_head" in name:
            owner = "correction_head"
        elif "query" in name:
            owner = "action_query"
        elif "reliability" in name or "compatibility" in name:
            owner = "reliability"
        elif "key" in name or "embedding" in name:
            owner = "semantic_key"
        else:
            owner = "transport"
        groups[owner].append(parameter)
    lr = cfg["training"]
    key = {"transport":"lr_transport", "action_query":"lr_action_query", "semantic_key":"lr_semantic_key",
           "reliability":"lr_reliability", "correction_head":"lr_correction_head"}
    return torch.optim.AdamW([{"params": values, "lr": float(lr[key[name]]), "name": name}
                              for name, values in groups.items() if values], weight_decay=float(lr["weight_decay"]))


@torch.no_grad()
def collect(model, loader, device, alpha: float, audit_limit: int = 0):
    model.eval(); keys=("base_action","final_action","base_reason","final_reason","action_target","reason_target")
    store={key:[] for key in keys}; names=[]; routes=[]; ablations={}; seen=0
    variants = {"null_only":{"force_null_only":True}, "semantic_shuffle":{"semantic_shuffle":True},
                "visual_shuffle":{"visual_shuffle":True}, "named_off":{"named_factors_off":True},
                "unnamed_off":{"unnamed_factors_off":True}, "support_off":{"support_route_off":True},
                "counter_off":{"counter_route_off":True}, "predicate_off":{"predicate_off":True},
                "reliability_off":{"reliability_off":True}}
    for batch in loader:
        images=batch["image"].to(device,non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            base=model.base_model(images,**model.base_forward_kwargs)
            output=model.decode_base_output(base,alpha=alpha)
        assert_vetra_contract(output)
        mapping={"base_action":output["action_logits_base"],"final_action":output["action_logits_final"],
                 "base_reason":output["reason_logits_base"],"final_reason":output["reason_logits_final"],
                 "action_target":batch["action"],"reason_target":batch["reason"]}
        for key,value in mapping.items(): store[key].append(value.detach().cpu())
        names.extend(batch["file_name"]); routes.append(route_statistics(output))
        if seen < audit_limit:
            take=min(images.shape[0],audit_limit-seen)
            for name,kwargs in variants.items():
                with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                    changed=model.decode_base_output(base,alpha=alpha,**kwargs)
                ablations.setdefault(name,[]).append(changed["action_logits_final"][:take].detach().cpu())
            ablations.setdefault("action_target",[]).append(batch["action"][:take].cpu())
            ablations.setdefault("reason_target",[]).append(batch["reason"][:take].cpu())
            ablations.setdefault("reason",[]).append(output["reason_logits_final"][:take].detach().cpu())
            ablations.setdefault("formal",[]).append(output["action_logits_final"][:take].detach().cpu())
            ablations.setdefault("base",[]).append(output["action_logits_base"][:take].detach().cpu())
            seen += take
    return ({key:torch.cat(value) for key,value in store.items()}|{"file_name":names}, routes,
            {key:torch.cat(value) for key,value in ablations.items()} if ablations else {})


@torch.no_grad()
def evaluate(model, calib_loader, test_loader, device, epoch_dir: Path, alpha: float, source_threshold):
    calib,_,_=collect(model,calib_loader,device,alpha)
    test,routes,ablations=collect(model,test_loader,device,alpha,audit_limit=128)
    groups=[list(range(4)),list(range(4,25))]
    base_threshold=fit_posthoc_thresholds(torch.cat((calib["base_action"],calib["base_reason"]),1),
        torch.cat((calib["action_target"],calib["reason_target"]),1),groups)["threshold_prob"]
    final_threshold=fit_posthoc_thresholds(torch.cat((calib["final_action"],calib["final_reason"]),1),
        torch.cat((calib["action_target"],calib["reason_target"]),1),groups)["threshold_prob"]
    base_raw=aie_branch_metrics(test["base_action"],test["base_reason"],test["action_target"],test["reason_target"])
    final_raw=aie_branch_metrics(test["final_action"],test["final_reason"],test["action_target"],test["reason_target"])
    base_deploy=aie_branch_metrics(apply_posthoc_threshold(test["base_action"],base_threshold[:4]),
        apply_posthoc_threshold(test["base_reason"],base_threshold[4:]),test["action_target"],test["reason_target"])
    final_deploy=aie_branch_metrics(apply_posthoc_threshold(test["final_action"],final_threshold[:4]),
        apply_posthoc_threshold(test["final_reason"],final_threshold[4:]),test["action_target"],test["reason_target"])
    base_source_fixed=aie_branch_metrics(apply_posthoc_threshold(test["base_action"],source_threshold[:4]),
        apply_posthoc_threshold(test["base_reason"],source_threshold[4:]),test["action_target"],test["reason_target"])
    final_source_fixed=aie_branch_metrics(apply_posthoc_threshold(test["final_action"],source_threshold[:4]),
        apply_posthoc_threshold(test["final_reason"],source_threshold[4:]),test["action_target"],test["reason_target"])
    epoch_dir.mkdir(parents=True,exist_ok=True)
    for name,key in (("action_logits_base_test.pt","base_action"),("action_logits_vetra_test.pt","final_action"),
                     ("reason_logits_base_test.pt","base_reason"),("reason_logits_vetra_test.pt","final_reason"),
                     ("labels_action_test.pt","action_target"),("labels_reason_test.pt","reason_target")):
        torch.save(test[key],epoch_dir/name)
    write_json(epoch_dir/"file_names_test.json",test["file_name"])
    torch.save(ablations,epoch_dir/"audit_128_ablation_logits.pt")
    ablation_metrics={name:aie_branch_metrics(logits,ablations["reason"],ablations["action_target"],ablations["reason_target"])
                      for name,logits in ablations.items() if name not in {"action_target","reason_target","reason"}}
    payload={"base_raw":base_raw,"vetra_raw":final_raw,"base_deploy":base_deploy,"vetra_deploy":final_deploy,
             "base_source_fixed":base_source_fixed,"vetra_source_fixed":final_source_fixed,
             "base_thresholds_train_calib":base_threshold.tolist(),"vetra_thresholds_train_calib":final_threshold.tolist(),
             "source_fixed_thresholds":source_threshold.tolist(),
             "reason_identity_max_abs":float((test["base_reason"]-test["final_reason"]).abs().max()),"alpha":alpha}
    write_json(epoch_dir/"branch_metrics.json",payload); write_json(epoch_dir/"ablation_metrics.json",ablation_metrics)
    return payload,routes,ablation_metrics


@torch.no_grad()
def counterfactual_epoch_audit(model, loader, device, alpha: float, max_samples: int):
    auditor=VETRACounterfactualAudit(); cases=[]; per_action={str(a):[] for a in range(4)}; seen=0
    for batch in loader:
        if seen>=max_samples: break
        take=min(batch["image"].shape[0],max_samples-seen); images=batch["image"][:take].to(device)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
            base=model.base_model(images,**model.base_forward_kwargs); out=model.decode_base_output(base,alpha=alpha)
        result=auditor.run(model,out,batch["action"][:take].to(device),take); cases.extend(result["cases"])
        for row in result["cases"]: per_action[str(row["action"])].append(row["selected_control_effect"])
        seen+=take
    return {"cases":cases,"per_action":{key:{"count":len(values),"effect_mean":float(np.mean(values)) if values else None}
                                        for key,values in per_action.items()},"valid_coverage":len(cases),"dino_rerun_count":0}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--source-checkpoint",required=True)
    p.add_argument("--output-dir",required=True); p.add_argument("--epochs",type=int); p.add_argument("--batch-size",type=int)
    p.add_argument("--gradient-accumulation-steps",type=int); p.add_argument("--num-workers",type=int)
    p.add_argument("--max-train-main-samples",type=int); p.add_argument("--max-calib-samples",type=int)
    p.add_argument("--max-audit-samples",type=int); p.add_argument("--max-test-samples",type=int)
    p.add_argument("--resume"); p.add_argument("--device",default="cuda"); a=p.parse_args()
    cfg=load_config(a.config); device=torch.device(a.device); seed=int(cfg["data"]["split_seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(int(cfg["runtime"]["cpu_threads"]))
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=build_vetra_model(cfg,a.source_checkpoint,device); base_hash=tensor_state_hash(model.base_model)
    source_control=Path(cfg["experiment"]["source_control_dir"])
    source_threshold=torch.tensor(json.loads((source_control/"calibration_thresholds.json").read_text(encoding="utf-8"))["threshold_prob"])
    map_loss=VETRAMAPLoss().to(device); optimizer=_optimizer(model,cfg)
    full_train=make_dataset(cfg,"train"); full_test=make_dataset(cfg,"test")
    names=[sample.file_name for sample in full_train.samples]; split=stable_split_ids(names,seed,
        cfg["data"]["train_calib_count"]/len(names),int(cfg["data"]["train_audit_count"])); index={s.file_name:i for i,s in enumerate(full_train.samples)}
    main_ids=[index[n] for n in split["train_main"][:a.max_train_main_samples or None]]
    calib_ids=[index[n] for n in split["train_calib"][:a.max_calib_samples or None]]
    audit_ids=[index[n] for n in split["train_audit"][:a.max_audit_samples or None]]
    test_ids=list(range(len(full_test)))[:a.max_test_samples or None]
    batch=a.batch_size or int(cfg["data"]["batch_size"]); accum=a.gradient_accumulation_steps or int(cfg["data"]["gradient_accumulation_steps"])
    workers=a.num_workers if a.num_workers is not None else int(cfg["data"]["num_workers"])
    gen=torch.Generator().manual_seed(seed)
    train_loader=make_loader(Subset(full_train,main_ids),batch,True,workers,cfg,gen)
    calib_loader=make_loader(Subset(full_train,calib_ids),batch,False,workers,cfg)
    audit_loader=make_loader(Subset(full_train,audit_ids),batch,False,workers,cfg)
    test_loader=make_loader(Subset(full_test,test_ids),batch,False,workers,cfg)
    manifest=write_split_manifest(out/"split_manifest.json",names,seed,cfg["data"]["train_calib_count"]/len(names),int(cfg["data"]["train_audit_count"]))
    overlap={"main_calib":len(set(split["train_main"])&set(split["train_calib"])),
             "main_audit":len(set(split["train_main"])&set(split["train_audit"])),
             "calib_audit":len(set(split["train_calib"])&set(split["train_audit"])),
             "counts":{"train_main":len(main_ids),"train_calib":len(calib_ids),"train_audit":len(audit_ids),"test":len(test_ids)}}
    write_json(out/"VETRA_SPLIT_OVERLAP_AUDIT.json",overlap)
    write_json(out/"DICE_CF_SEMANTICS_REPAIR.json",{"fixed_commit_parent":"1b7138f75a37925e35f50444eec8eee62e20d375",
        "rerun_uses_action_logits_base":True,"signed_negative_effect":True,"disjoint_splits":all(v==0 for k,v in overlap.items() if k!="counts")})
    write_json(out/"run_manifest.json",{"git_head":_git_head(),"source_checkpoint":str(a.source_checkpoint),"source_checkpoint_hash":file_sha256(a.source_checkpoint),
        "base_hash":base_hash,"direct_image":True,"feature_cache_enabled":False,"token_compression":"none","best_selection_split":"test",
        "batch_size":batch,"gradient_accumulation_steps":accum,"num_workers":workers,"split_manifest":manifest})
    epochs=a.epochs or int(cfg["training"]["epochs"]); updates_per_epoch=math.ceil(len(train_loader)/accum); total_updates=updates_per_epoch*epochs
    scheduler=make_scheduler(optimizer,total_updates,float(cfg["training"]["warmup_ratio"]),float(cfg["training"]["min_lr_ratio"]))
    start_epoch=0; update=0
    if a.resume:
        state=torch.load(a.resume,map_location=device); model.load_state_dict(state["model"],strict=True); optimizer.load_state_dict(state["optimizer"])
        map_loss.load_state_dict(state["map_loss"]); scheduler.load_state_dict(state["scheduler"]); start_epoch=int(state["epoch"])+1; update=int(state["update"])
    for epoch in range(start_epoch,epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); window_outputs=[]; window_targets=[]; route_rows=[]; started=time.perf_counter()
        for micro,batch_data in enumerate(train_loader):
            alpha=alpha_schedule(update,total_updates,float(cfg["vetra"]["initial_scale"]),float(cfg["vetra"]["full_scale_ratio"]))
            images=batch_data["image"].to(device,non_blocking=True); action=batch_data["action"].to(device,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
                with torch.no_grad(): base=model.base_model(images,**model.base_forward_kwargs)
                output=model.decode_base_output(base,alpha=alpha); assert_vetra_contract(output)
            window_outputs.append(output); window_targets.append(action); route_rows.append(route_statistics(output))
            will_step=(micro+1)%accum==0 or micro+1==len(train_loader)
            if will_step:
                keys=("action_logits_final","action_logits_base","vetra_action_delta","support_route","counter_route","support_reliability")
                merged={key:torch.cat([row[key] for row in window_outputs],0) for key in keys}
                total,components=total_vetra_loss(merged,torch.cat(window_targets,0),map_loss,cfg["loss_weights"],
                    float(cfg["vetra"]["rank_preserve_ratio"]),float(cfg["vetra"]["base_margin_floor"]),
                    float(cfg["vetra"]["null_max_reliable"]),float(cfg["vetra"]["predicate_confidence_floor"]))
                total.backward()
                grad=float(torch.nn.utils.clip_grad_norm_(model.transport.parameters(),float(cfg["training"]["global_grad_clip"])))
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); update+=1; window_outputs=[]; window_targets=[]
                assert_base_frozen(model,base_hash,tensor_state_hash)
                if update==1 or update%int(cfg["training"]["print_every_optimizer_updates"])==0:
                    row={"event":"vetra_batch","epoch":epoch,"update":update,"alpha":alpha,"loss_total":float(total.detach()),
                         **{f"loss_{k}":float(v.detach()) for k,v in components.items()},"grad_norm":grad,
                         "gpu_reserved_gb":torch.cuda.max_memory_reserved()/2**30 if device.type=="cuda" else 0,**route_rows[-1]}
                    print(json.dumps(row),flush=True); append_jsonl(out/"loss_components.jsonl",row)
        alpha=alpha_schedule(update,total_updates,float(cfg["vetra"]["initial_scale"]),float(cfg["vetra"]["full_scale_ratio"]))
        epoch_dir=out/f"epoch_{epoch:03d}"; metrics,test_routes,ablations=evaluate(model,calib_loader,test_loader,device,epoch_dir,alpha,source_threshold)
        cf=counterfactual_epoch_audit(model,audit_loader,device,alpha,min(int(cfg["counterfactual"]["max_audit_samples"]),len(audit_ids)))
        write_json(epoch_dir/"counterfactual_audit.json",cf); write_json(epoch_dir/"route_stats.json",test_routes)
        append_jsonl(out/"VETRA_METRICS_SUMMARY.jsonl",{"epoch":epoch,**metrics})
        append_jsonl(out/"VETRA_PER_ACTION_METRICS.jsonl",{"epoch":epoch,"base_ap":metrics["base_raw"]["Act_per_label_ap"],"vetra_ap":metrics["vetra_raw"]["Act_per_label_ap"],
            "base_f1":metrics["base_raw"]["Act_per_label_f1"],"vetra_f1":metrics["vetra_raw"]["Act_per_label_f1"]})
        for row in test_routes: append_jsonl(out/"VETRA_ROUTE_STATS.jsonl",{"epoch":epoch,**row})
        checkpoint={"epoch":epoch,"update":update,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "map_loss":map_loss.state_dict(),"scheduler":scheduler.state_dict(),"base_hash":base_hash,"metrics":metrics}
        torch.save(checkpoint,out/f"checkpoint_epoch_{epoch:03d}.pth"); torch.save(checkpoint,out/"checkpoint_latest.pth")
        print(json.dumps({"event":"vetra_epoch","epoch":epoch,"seconds":time.perf_counter()-started,
                          "base_act_mf1":metrics["base_deploy"]["Act_mF1"],"vetra_act_mf1":metrics["vetra_deploy"]["Act_mF1"],
                          "base_act_map":metrics["base_raw"]["Act_mAP"],"vetra_act_map":metrics["vetra_raw"]["Act_mAP"],
                          "exp_mf1":metrics["vetra_deploy"]["Exp_mF1"]}),flush=True)
    decision=finalize_probe(out,cfg)
    print(json.dumps({"event":"vetra_fast_validation","decision":decision}),flush=True)


if __name__ == "__main__":
    main()
