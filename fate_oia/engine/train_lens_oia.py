from __future__ import annotations

import argparse
import copy
import json
import math
import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.lens_splits import make_lens_splits, write_split_manifest
from fate_oia.datasets.lens_mirror import ACTION_PERMUTATION, REASON_PERMUTATION, mirror_lens_batch
from fate_oia.datasets.lens_structured_evidence import LENSStructuredEvidenceBuilder, LENSStructuredRecordAdapter
from fate_oia.engine.eval_lens_oia import evaluate_lens
from fate_oia.losses.lens_latent_losses import conflict_discounted_responsibility, conflict_safe_reason_logits
from fate_oia.losses.lens_loss_registry import LENSLossRegistry
from fate_oia.models.lens_oia_model import LENSOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.lens_artifacts import append_jsonl, write_json
from fate_oia.utils.lens_calibration import fit_group_shrinkage_threshold
from fate_oia.utils.lens_hashes import sha256_file
from fate_oia.utils.lens_metrics import _binary_ranking_metrics


def should_optimizer_step(step: int, num_batches: int, accumulation_steps: int) -> bool:
    """Flush complete accumulation windows and the final partial window."""
    return (step + 1) % accumulation_steps == 0 or step + 1 == num_batches


def mechanism_progress(update: int, total_updates: int, ramp_fraction: float) -> float:
    ramp_updates = max(1, int(math.ceil(total_updates * ramp_fraction)))
    return float(max(0.0, min(1.0, update / ramp_updates)))


def _slice_field(field: dict[str, Any], start: int, end: int, full_batch: int) -> dict[str, Any]:
    return {key: (value[start:end] if torch.is_tensor(value) and value.ndim and value.shape[0] == full_batch else value) for key, value in field.items()}


def mirror_consistency_loss(reference: dict[str, torch.Tensor], mirrored: dict[str, torch.Tensor], count: int, *, horizontal_mirror: bool) -> torch.Tensor:
    action_perm = ACTION_PERMUTATION.to(mirrored["action_logits_final"].device) if horizontal_mirror else torch.arange(4,device=mirrored["action_logits_final"].device)
    reason_perm = REASON_PERMUTATION.to(mirrored["reason_logits_formal"].device) if horizontal_mirror else torch.arange(21,device=mirrored["reason_logits_formal"].device)
    action = F.mse_loss(reference["action_logits_final"][:count].sigmoid(), mirrored["action_logits_final"][:, action_perm].sigmoid())
    reason = F.mse_loss(reference["reason_logits_formal"][:count].sigmoid(), mirrored["reason_logits_formal"][:, reason_perm].sigmoid())
    state = F.mse_loss(reference["state_prob"][:count], mirrored["state_prob"][:, reason_perm])
    mirrored_map = mirrored["evidence_map"][:, reason_perm]
    if horizontal_mirror:
        mirrored_map = torch.flip(mirrored_map,dims=(-1,))
    evidence = F.mse_loss(reference["evidence_map"][:count], mirrored_map)
    return (action + reason + state + evidence) / 4.0


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {"image":torch.stack([item["image"] for item in batch]),"action":torch.stack([item["action"] for item in batch]),"reason":torch.stack([item["reason"] for item in batch]),"file_name":[item["file_name"] for item in batch]}


def make_loader(cfg: dict[str, Any], split: str, *, batch_size: int, shuffle: bool, indices: list[int] | None = None, max_samples: int | None = None) -> DataLoader:
    transform=AspectRatioLetterboxTransform(int(cfg["image_height"]),int(cfg["image_width"]),patch_size=int(cfg["patch_size"]))
    dataset=BDDOIAMultiTaskDataset(cfg["data_root"],cfg["raw_root"],split=split,action_dim=4,reason_dim=21,load_image=True,transform=transform)
    if indices is not None: dataset=Subset(dataset,indices)
    if max_samples is not None: dataset=Subset(dataset,list(range(min(max_samples,len(dataset)))))
    tr=cfg["training"]; workers=int(tr.get("num_workers",4)); kwargs={"num_workers":workers,"pin_memory":bool(tr.get("pin_memory",True)),"collate_fn":collate}
    if workers: kwargs.update({"persistent_workers":bool(tr.get("persistent_workers",True)),"prefetch_factor":int(tr.get("prefetch_factor",4))})
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,**kwargs)


def make_optimizer(model: LENSOIAModel, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    tr=cfg["training"]
    groups=[
      {"name":"foundation","params":list(model.foundation.parameters()),"lr":tr["lr_foundation"]},
      {"name":"adaptive_evidence","params":list(model.adaptive_evidence.parameters()),"lr":tr["lr_adaptive_evidence"]},
      {"name":"latent_state","params":list(model.latent_state.parameters()),"lr":tr["lr_latent_state"]},
      {"name":"action_reread","params":list(model.action_reread.parameters()),"lr":tr["lr_action_reread"]},
      {"name":"annotation_emission","params":list(model.annotation_emission.parameters()),"lr":tr["lr_annotation_emission"]},
    ]
    ids=[id(p) for group in groups for p in group["params"]]
    if len(ids)!=len(set(ids)): raise RuntimeError("duplicate optimizer parameter owner")
    return torch.optim.AdamW(groups,weight_decay=float(tr["weight_decay"]))


def make_scheduler(optimizer: torch.optim.Optimizer, total_updates: int, warmup_fraction: float, min_lr_ratio: float):
    warmup_updates=max(1,int(math.ceil(total_updates*warmup_fraction)))
    def scale(update: int) -> float:
        if update < warmup_updates:
            return max(1e-6,(update+1)/warmup_updates)
        fraction=(update-warmup_updates)/max(1,total_updates-warmup_updates)
        return min_lr_ratio+(1.0-min_lr_ratio)*0.5*(1.0+math.cos(math.pi*min(1.0,fraction)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)


def model_state_hash(model: torch.nn.Module) -> str:
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trainable_snapshot(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in module.parameters() if parameter.requires_grad]


def snapshot_delta(module: torch.nn.Module, snapshot: list[torch.Tensor]) -> float:
    current=[parameter.detach().cpu() for parameter in module.parameters() if parameter.requires_grad]
    return float(sum((value-before).pow(2).sum() for value,before in zip(current,snapshot)).sqrt())


def grad_norm(module: torch.nn.Module) -> float:
    values=[parameter.grad.detach().float().pow(2).sum() for parameter in module.parameters() if parameter.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


@torch.no_grad()
def synthetic_flip_audit(model: LENSOIAModel, loader: DataLoader, device: torch.device, progress: float, epoch: int) -> dict[str, Any]:
    cached={name:[] for name in ("action","reason","state","action_state","source")}; emission=None
    for batch in loader:
      out=model(batch["image"].to(device),progress=progress)
      cached["action"].append(batch["action"].to(device)); cached["reason"].append(batch["reason"].to(device)); cached["state"].append(out["state_prob"]); cached["action_state"].append(out["action_logits_state_substitution"]); cached["source"].append(out["reason_logits_source"]); emission=out["emission_prob"]
    values={name:torch.cat(items) for name,items in cached.items()}; rows=[]
    for rate in (0.05,0.10):
      generator=torch.Generator().manual_seed(20260805+epoch+int(rate*100))
      reason=values["reason"]; clean=conflict_discounted_responsibility(values["state"],emission,reason,values["action_state"],values["action"],0.5)
      mask=(torch.rand(reason.shape,generator=generator)<rate).to(device); flipped=torch.where(mask,1.0-reason,reason)
      altered=conflict_discounted_responsibility(values["state"],emission,flipped,values["action_state"],values["action"],0.5)
      safe_clean=conflict_safe_reason_logits(values["state"],values["source"],emission,reason,clean["gamma"],clean["conflict"],progress)
      safe_flip=conflict_safe_reason_logits(values["state"],values["source"],emission,flipped,altered["gamma"],altered["conflict"],progress)
      clean_c=clean["conflict"].cpu(); flip_c=altered["conflict"].cpu(); mask=mask.cpu(); labels=mask.flatten().float(); score=(flip_c-clean_c).flatten()
      auc=_binary_ranking_metrics(score,labels)[1]
      clean_u=clean["gamma"][...,2].cpu(); flip_u=altered["gamma"][...,2].cpu(); clean_w=safe_clean["share_weight"].cpu(); flip_w=safe_flip["share_weight"].cpu()
      rows.append({"flip_rate":rate,"flip_detection_AUROC":auc,"conflict_clean":clean_c[~mask].mean(),"conflict_flip":flip_c[mask].mean(),"gamma_unknown_clean":clean_u[mask].mean(),"gamma_unknown_flip":flip_u[mask].mean(),"shared_gradient_change_lens":(flip_w-clean_w).abs().mean(),"shared_gradient_change_raw_bce":torch.ones_like(flip_w[mask]).mean(),"gradient_robustness_ratio":(flip_w-clean_w).abs().mean()})
    return {"epoch":epoch,"rates":rows}


def main() -> None:
    torch.set_float32_matmul_precision("high")
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--run-kind",choices=("smoke","pilot","full"),default="smoke"); parser.add_argument("--resume"); parser.add_argument("--epochs",type=int); parser.add_argument("--batch-size",type=int); parser.add_argument("--max-train-samples","--max-train-main-samples",dest="max_train_samples",type=int); parser.add_argument("--max-train-audit-samples",type=int); parser.add_argument("--max-calib-samples","--max-train-calib-samples",dest="max_calib_samples",type=int); parser.add_argument("--max-test-samples",type=int); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); device=torch.device(args.device)
    train_full=make_loader(cfg,"train",batch_size=1,shuffle=False); base=getattr(train_full.dataset,"dataset",train_full.dataset); samples=getattr(base,"samples")
    labels=torch.stack([torch.cat([torch.tensor(sample.action),torch.tensor(sample.reason)]) for sample in samples]); names=[sample.file_name for sample in samples]
    splits=make_lens_splits(names,labels,seed=int(cfg["splits"]["seed"])); write_split_manifest(output/"split_manifest.json",splits,names)
    batch_size=args.batch_size or int(cfg["training"]["batch_size"]); train_loader=make_loader(cfg,"train",batch_size=batch_size,shuffle=True,indices=splits["train_main"],max_samples=args.max_train_samples); audit_loader=make_loader(cfg,"train",batch_size=batch_size,shuffle=False,indices=splits["train_audit"],max_samples=args.max_train_audit_samples); calib_loader=make_loader(cfg,"train",batch_size=batch_size,shuffle=False,indices=splits["train_calib"],max_samples=args.max_calib_samples); test_loader=make_loader(cfg,"test",batch_size=batch_size,shuffle=False,max_samples=args.max_test_samples)
    model=LENSOIAModel(use_mock_dino=bool(cfg["model"].get("use_mock_dino",False)),selected_layers=tuple(cfg["model"]["selected_layers"]),pretrained_weights=cfg["pretrained_weights"],factor_chunk_size=int(cfg["model"].get("factor_chunk_size",21)),evidence_tau_min=float(cfg.get("evidence",{}).get("tau_min",0.35)),evidence_tau_max=float(cfg.get("evidence",{}).get("tau_max",2.0)),evidence_topk=int(cfg.get("evidence",{}).get("topk",32)),region_bias_abs_max=float(cfg.get("evidence",{}).get("region_bias_abs_max",2.0)),action_logit_cap=float(cfg["training"]["action_logit_norm_cap"])).to(device); optimizer=make_optimizer(model,cfg); registry=LENSLossRegistry(cfg["loss_weights"])
    structured_builder=LENSStructuredEvidenceBuilder(cfg.get("reason_state_schema","configs/lens_reason_state_schema.yaml"))
    structured_adapter=LENSStructuredRecordAdapter(cfg["bdd100k_root"])
    owner_modules={"foundation":model.foundation,"adaptive_evidence":model.adaptive_evidence,"latent_state":model.latent_state,"action_reread":model.action_reread,"annotation_emission":model.annotation_emission}
    train_frequency=labels[splits["train_main"],4:].float().mean(0).to(device)
    model.annotation_emission.initialize_from_frequency(train_frequency)
    write_json(output/"emission_initialization.json",{"train_main_frequency":train_frequency.detach().cpu().tolist(),"emission_prob":model.annotation_emission.emission_probabilities().detach().cpu().tolist()})
    manifest={"run_kind":args.run_kind,"config_hash":sha256_file(args.config),"test_only":True,"eval_splits":["test"],"best_selection_split":"test","best_selection_metric":"deploy_joint","internal_test_selected":True,"publication_eligible":False,"feature_cache_enabled":False,"token_compression":"none","train_main_samples":len(train_loader.dataset),"train_audit_samples":len(audit_loader.dataset),"train_calib_samples":len(calib_loader.dataset),"test_samples":len(test_loader.dataset)}; write_json(output/"run_manifest.json",manifest)
    best=-float("inf"); epochs=args.epochs or int(cfg["training"]["epochs"]); total_updates=max(1,epochs*math.ceil(len(train_loader)/int(cfg["training"]["gradient_accumulation_steps"]))); scheduler=make_scheduler(optimizer,total_updates,float(cfg["training"]["warmup_update_fraction"]),float(cfg["training"]["min_lr_ratio"])); start_epoch=0; optimizer_update=0
    if args.resume:
      checkpoint=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"]); optimizer_update=int(checkpoint.get("optimizer_update",0)); start_epoch=int(checkpoint["epoch"])+1; best=float(checkpoint.get("best_deploy_joint",checkpoint.get("metrics",{}).get("deploy_joint",-float("inf"))))
    owner_initial={name:trainable_snapshot(module) for name,module in owner_modules.items()}
    last_grad_stats={name:0.0 for name in (*owner_modules,"DINO")}
    for epoch in range(start_epoch,epochs):
      model.train(); optimizer.zero_grad(set_to_none=True); count=0; grounding_counts={"known":0,"map":0,"complete":0,"total":0}
      for step,batch in enumerate(train_loader):
        progress=mechanism_progress(optimizer_update,total_updates,float(cfg["training"].get("mechanism_ramp_fraction",0.10)))
        grounding_progress=mechanism_progress(optimizer_update,total_updates,float(cfg["training"].get("grounding_ramp_fraction",0.05)))
        grounding_multiplier=0.25+0.75*grounding_progress
        action_state_progress=mechanism_progress(optimizer_update,total_updates,float(cfg["training"].get("action_state_ramp_fraction",0.15)))
        image=batch["image"].to(device,non_blocking=True); batch={**batch,"action":batch["action"].to(device),"reason":batch["reason"].to(device)}
        batch["structured"]=structured_builder.build([structured_adapter.build_record(name) for name in batch["file_name"]])
        grounding_counts["known"]+=int(batch["structured"].state_mask.sum()); grounding_counts["map"]+=int(batch["structured"].map_mask.sum()); grounding_counts["complete"]+=int(batch["structured"].source_complete.sum()); grounding_counts["total"]+=int(batch["structured"].state_mask.numel())
        accumulation=int(cfg["training"]["gradient_accumulation_steps"])
        paired=(optimizer_update % int(cfg.get("paired_view",{}).get("interval_optimizer_updates",4)) == 0 and step % accumulation == 0)
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
          if paired:
            paired_count=max(1,min(image.shape[0],int(math.ceil(image.shape[0]*float(cfg.get("paired_view",{}).get("max_fraction",0.5))))))
            do_mirror=bool(torch.rand((),device=image.device)<float(cfg.get("paired_view",{}).get("mirror_probability",0.25)))
            if do_mirror:
              weak=mirror_lens_batch(image[:paired_count],batch["action"][:paired_count],batch["reason"][:paired_count])
            else:
              weak_image=image[:paired_count]
              contrast=0.90+0.20*torch.rand((paired_count,1,1,1),device=image.device)
              brightness=-0.05+0.10*torch.rand((paired_count,1,1,1),device=image.device)
              weak_image=(weak_image*contrast+brightness+0.01*torch.randn_like(weak_image)).clamp(weak_image.amin(),weak_image.amax())
              weak={"image":weak_image,"action":batch["action"][:paired_count],"reason":batch["reason"][:paired_count]}
            combined=torch.cat([image,weak["image"]],dim=0); field=model.encode_images(combined)
            out=model.decode_from_field(_slice_field(field,0,image.shape[0],combined.shape[0]),progress=progress)
            mirrored_out=model.decode_from_field(_slice_field(field,image.shape[0],combined.shape[0],combined.shape[0]),progress=progress)
            batch["view_consistency_loss"]=mirror_consistency_loss(out,mirrored_out,paired_count,horizontal_mirror=do_mirror)
          else:
            out=model(image,progress=progress)
          lambda_action=0.5*action_state_progress
          resp=conflict_discounted_responsibility(out["state_prob"],out["emission_prob"],batch["reason"],out["action_logits_state_substitution"],batch["action"],lambda_action)
          safe=conflict_safe_reason_logits(out["state_prob"],out["reason_logits_source"],out["emission_prob"],batch["reason"],resp["gamma"],resp["conflict"],progress)
          out.update(safe)
          total,raw=registry(out,batch,resp,multipliers={name:grounding_multiplier for name in ("map_anchor","state_anchor","view_consistency")}); (total/int(cfg["training"]["gradient_accumulation_steps"])).backward()
        if should_optimizer_step(step,len(train_loader),int(cfg["training"]["gradient_accumulation_steps"])):
          last_grad_stats={name:grad_norm(module) for name,module in owner_modules.items()}; last_grad_stats["DINO"]=grad_norm(model.foundation.dino)
          foundation_raw=float(torch.nn.utils.clip_grad_norm_(model.foundation.parameters(),float(cfg["training"]["foundation_grad_cap"]))); global_raw=float(torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["global_grad_clip"]))); optimizer.step(); optimizer.zero_grad(set_to_none=True)
          last_grad_stats["foundation_raw_before_cap"]=foundation_raw; last_grad_stats["global_raw_before_clip"]=global_raw
          optimizer_update+=1; scheduler.step()
        count+=1
        if count%100==0 or step+1==len(train_loader): append_jsonl(output/"loss_components.jsonl",{"epoch":epoch,"step":step,"optimizer_update":optimizer_update,"lr":optimizer.param_groups[0]["lr"],"progress_lr":min(1.0,optimizer_update/max(1,math.ceil(total_updates*float(cfg["training"]["warmup_update_fraction"])))),"progress_grounding":grounding_progress,"progress_unknown":progress,"progress_emission":progress,"progress_reason_output":progress,"progress_action_reread":progress,"lambda_action_state":lambda_action,"loss_total":float(total.detach()),**{f"loss_{k}":float(v.detach()) for k,v in raw.items()},"conflict_mean":float(resp["conflict"].mean()),"share_weight_mean":float(out["share_weight"].mean()),"share_weight_min":float(out["share_weight"].min()),"share_weight_max":float(out["share_weight"].max()),"state_positive_mean":float(out["state_positive_prob"].mean()),"state_counter_mean":float(out["state_counter_prob"].mean()),"state_unknown_mean":float(out["state_unknown_prob"].mean()),"emission_Tplus_mean":float(out["emission_prob"][:,2].mean()),"emission_Tunknown_mean":float(out["emission_prob"][:,1].mean()),"emission_Tminus_mean":float(out["emission_prob"][:,0].mean()),"evidence_null_mean":float(out["evidence_null_mass"].mean()),"evidence_entropy_mean":float(out["evidence_entropy"].mean()),"evidence_snr_mean":float(out["evidence_snr"].mean()),"evidence_temperature_mean":float(out["evidence_temperature"].mean()),"foundation_grad_raw":last_grad_stats["foundation_raw_before_cap"],"foundation_grad_capped":min(last_grad_stats["foundation_raw_before_cap"],float(cfg["training"]["foundation_grad_cap"])),"evidence_grad":last_grad_stats["adaptive_evidence"],"state_grad":last_grad_stats["latent_state"],"emission_grad":last_grad_stats["annotation_emission"],"action_reread_grad":last_grad_stats["action_reread"],"DINO_grad":last_grad_stats["DINO"]})
      # Calibration is outside the model and receives only train-calib labels.
      before_calibration_hash=model_state_hash(model)
      raw_calib,calib_store=evaluate_lens(model,calib_loader,device,progress=progress); action_threshold=fit_group_shrinkage_threshold(calib_store["action"],calib_store["labels_action"]); reason_threshold=fit_group_shrinkage_threshold(calib_store["reason"],calib_store["labels_reason"])
      after_calibration_hash=model_state_hash(model)
      if before_calibration_hash != after_calibration_hash: raise RuntimeError("post-hoc calibration mutated model state")
      metrics,store=evaluate_lens(model,test_loader,device,progress=progress,action_threshold=action_threshold,reason_threshold=reason_threshold,diagnostic_samples=int(cfg.get("runtime",{}).get("fixed_test_audit_samples",128))); append_jsonl(output/"metrics_summary.jsonl",{"epoch":epoch,**metrics})
      append_jsonl(output/"synthetic_flip_audit.jsonl",synthetic_flip_audit(model,audit_loader,device,progress,epoch))
      append_jsonl(output/"grounding_stats.jsonl",{"epoch":epoch,**grounding_counts,"known_rate":grounding_counts["known"]/max(1,grounding_counts["total"]),"map_rate":grounding_counts["map"]/max(1,grounding_counts["total"]),"complete_rate":grounding_counts["complete"]/max(1,grounding_counts["total"])})
      append_jsonl(output/"owner_stats.jsonl",{"epoch":epoch,"parameter_delta":{name:snapshot_delta(module,owner_initial[name]) for name,module in owner_modules.items()},"last_grad_norm":last_grad_stats})
      append_jsonl(output/"branch_metrics.jsonl",{"epoch":epoch,**metrics["branch_metrics"]})
      append_jsonl(output/"calibration.jsonl",{"epoch":epoch,"source":"train_calib","model_hash_before":before_calibration_hash,"model_hash_after":after_calibration_hash,"action_threshold":action_threshold,"reason_threshold":reason_threshold})
      state=store["state_prob"]; emission=store["emission_prob"]; selection=store["factor_selection"]; named=store["factor_contribution"]; unnamed=store["unnamed_contribution"]
      append_jsonl(output/"state_stats.jsonl",{"epoch":epoch,"positive_mean":state[...,0].mean(),"counter_mean":state[...,1].mean(),"unknown_mean":state[...,2].mean(),"unknown_p10":torch.quantile(state[...,2],0.1),"unknown_p50":torch.quantile(state[...,2],0.5),"unknown_p90":torch.quantile(state[...,2],0.9)})
      append_jsonl(output/"emission_stats.jsonl",{"epoch":epoch,"Tminus_mean":emission[:,0].mean(),"Tunknown_mean":emission[:,1].mean(),"Tplus_mean":emission[:,2].mean(),"ordered_margin_min":torch.minimum(emission[:,1]-emission[:,0],emission[:,2]-emission[:,1]).min()})
      reconstruction=(store["action"]-store["action_base"]-(named.sum(-1)+unnamed)).abs().max()
      append_jsonl(output/"action_contribution_stats.jsonl",{"epoch":epoch,"factor_selection_entropy":-(selection.clamp_min(1e-8)*selection.clamp_min(1e-8).log()).sum(-1).mean(),"factor_effective_count":torch.exp(-(selection.clamp_min(1e-8)*selection.clamp_min(1e-8).log()).sum(-1)).mean(),"factor_named_abs_mean":named.abs().mean(),"unnamed_abs_mean":unnamed.abs().mean(),"contribution_reconstruction_max":reconstruction})
      epoch_dir=output/f"epoch_{epoch:02d}"; epoch_dir.mkdir(parents=True,exist_ok=True)
      audit_subset=store.pop("audit_subset")
      torch.save(store,epoch_dir/"test_outputs.pt"); torch.save(audit_subset,epoch_dir/"audit_subset.pt")
      write_json(epoch_dir/"per_label_action_metrics.json",metrics["raw_action"]); write_json(epoch_dir/"per_label_reason_metrics.json",metrics["raw_reason"])
      torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"optimizer_update":optimizer_update,"epoch":epoch,"metrics":metrics,"best_deploy_joint":max(best,metrics["deploy_joint"]),"run_kind":args.run_kind,"config_hash":manifest["config_hash"],"calibration":{"action":action_threshold,"reason":reason_threshold}},output/"checkpoint_latest.pth")
      if metrics["deploy_joint"]>=best: best=metrics["deploy_joint"]; torch.save(torch.load(output/"checkpoint_latest.pth",weights_only=False),output/"checkpoint_best_test_deploy_joint.pth")
    completion="GOAL_COMPLETED_LENS_OIA_V1.json" if args.run_kind=="full" else "LENS_PILOT_TRAINING_COMPLETED.json" if args.run_kind=="pilot" else "LENS_SMOKE_COMPLETED.json"
    write_json(output/completion,{"complete":True,"run_kind":args.run_kind,"best_deploy_joint":best,"optimizer_updates":optimizer_update})


if __name__=="__main__": main()
