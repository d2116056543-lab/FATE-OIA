from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.lens_splits import make_lens_splits, write_split_manifest
from fate_oia.engine.eval_lens_oia import evaluate_lens
from fate_oia.losses.lens_latent_losses import conflict_discounted_responsibility
from fate_oia.losses.lens_loss_registry import LENSLossRegistry
from fate_oia.models.lens_oia_model import LENSOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.lens_artifacts import append_jsonl, write_json
from fate_oia.utils.lens_calibration import fit_group_shrinkage_threshold
from fate_oia.utils.lens_hashes import sha256_file


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
      {"name":"foundation","params":model.foundation.parameters(),"lr":tr["lr_foundation"]},
      {"name":"adaptive_evidence","params":model.adaptive_evidence.parameters(),"lr":tr["lr_adaptive_evidence"]},
      {"name":"latent_state","params":model.latent_state.parameters(),"lr":tr["lr_latent_state"]},
      {"name":"action_reread","params":model.action_reread.parameters(),"lr":tr["lr_action_reread"]},
      {"name":"annotation_emission","params":model.annotation_emission.parameters(),"lr":tr["lr_annotation_emission"]},
    ]
    ids=[id(p) for group in groups for p in group["params"]]
    if len(ids)!=len(set(ids)): raise RuntimeError("duplicate optimizer parameter owner")
    return torch.optim.AdamW(groups,weight_decay=float(tr["weight_decay"]))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output-dir",required=True); parser.add_argument("--epochs",type=int); parser.add_argument("--batch-size",type=int); parser.add_argument("--max-train-samples",type=int); parser.add_argument("--max-test-samples",type=int); parser.add_argument("--device",default="cuda"); args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); device=torch.device(args.device)
    train_full=make_loader(cfg,"train",batch_size=1,shuffle=False); base=getattr(train_full.dataset,"dataset",train_full.dataset); samples=getattr(base,"samples")
    labels=torch.stack([torch.cat([torch.tensor(sample.action),torch.tensor(sample.reason)]) for sample in samples]); names=[sample.file_name for sample in samples]
    splits=make_lens_splits(names,labels,seed=int(cfg["splits"]["seed"])); write_split_manifest(output/"split_manifest.json",splits,names)
    batch_size=args.batch_size or int(cfg["training"]["batch_size"]); train_loader=make_loader(cfg,"train",batch_size=batch_size,shuffle=True,indices=splits["train_main"],max_samples=args.max_train_samples); calib_loader=make_loader(cfg,"train",batch_size=batch_size,shuffle=False,indices=splits["train_calib"]); test_loader=make_loader(cfg,"test",batch_size=batch_size,shuffle=False,max_samples=args.max_test_samples)
    model=LENSOIAModel(use_mock_dino=bool(cfg["model"].get("use_mock_dino",False))).to(device); optimizer=make_optimizer(model,cfg); registry=LENSLossRegistry(cfg["loss_weights"])
    manifest={"config_hash":sha256_file(args.config),"test_only":True,"eval_splits":["test"],"best_selection_split":"test","best_selection_metric":"deploy_joint","internal_test_selected":True,"publication_eligible":False,"feature_cache_enabled":False,"token_compression":"none"}; write_json(output/"run_manifest.json",manifest)
    best=-float("inf"); epochs=args.epochs or int(cfg["training"]["epochs"]); total_updates=max(1,epochs*math.ceil(len(train_loader)/int(cfg["training"]["gradient_accumulation_steps"])))
    for epoch in range(epochs):
      model.train(); optimizer.zero_grad(set_to_none=True); progress=min(1.0,(epoch+1)/max(epochs*0.10,1)); count=0
      for step,batch in enumerate(train_loader):
        image=batch["image"].to(device,non_blocking=True); batch={**batch,"action":batch["action"].to(device),"reason":batch["reason"].to(device)}
        with torch.autocast("cuda",dtype=torch.bfloat16,enabled=device.type=="cuda"):
          out=model(image,progress=progress); lambda_action=min(0.5,(epoch+1)/max(epochs*0.15,1)*0.5)
          resp=conflict_discounted_responsibility(out["state_prob"],out["emission_prob"],batch["reason"],out["action_logits_state_substitution"],batch["action"],lambda_action)
          total,raw=registry(out,batch,resp); (total/int(cfg["training"]["gradient_accumulation_steps"])).backward()
        if (step+1)%int(cfg["training"]["gradient_accumulation_steps"])==0:
          torch.nn.utils.clip_grad_norm_(model.foundation.parameters(),float(cfg["training"]["foundation_grad_cap"])); torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["training"]["global_grad_clip"])); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        count+=1
        if count%100==0: append_jsonl(output/"loss_components.jsonl",{"epoch":epoch,"step":step,"loss_total":float(total.detach()),**{f"loss_{k}":float(v.detach()) for k,v in raw.items()},"conflict_mean":float(resp["conflict"].mean()),"state_unknown_mean":float(out["state_unknown_prob"].mean()),"evidence_null_mean":float(out["evidence_null_mass"].mean())})
      # Calibration is outside the model and receives only train-calib labels.
      raw_calib,calib_store=evaluate_lens(model,calib_loader,device,progress=progress); action_threshold=fit_group_shrinkage_threshold(calib_store["action"],calib_store["labels_action"]); reason_threshold=fit_group_shrinkage_threshold(calib_store["reason"],calib_store["labels_reason"])
      metrics,store=evaluate_lens(model,test_loader,device,progress=progress,action_threshold=action_threshold,reason_threshold=reason_threshold); append_jsonl(output/"metrics_summary.jsonl",{"epoch":epoch,**metrics})
      torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"epoch":epoch,"metrics":metrics,"calibration":{"action":action_threshold,"reason":reason_threshold}},output/"checkpoint_latest.pth")
      if metrics["deploy_joint"]>=best: best=metrics["deploy_joint"]; torch.save(torch.load(output/"checkpoint_latest.pth",weights_only=False),output/"checkpoint_best_test_deploy_joint.pth")
    write_json(output/"GOAL_COMPLETED_LENS_OIA_V1.json",{"complete":True,"best_deploy_joint":best})


if __name__=="__main__": main()
