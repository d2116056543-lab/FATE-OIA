from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fate_oia.datasets.coev_video_dataset import CoEVVideoDataset, coev_collate
from fate_oia.engine.evaluate_coev_oia import evaluate, final_intervention_diagnostics, is_better
from fate_oia.losses.coev_losses import CoEVLoss
from fate_oia.models.coev_oia_model import CoEVOIAModel
from fate_oia.utils.coev_checkpoint import atomic_torch_save, capture_rng, restore_rng, runtime_identity, verify_resume_identity
from fate_oia.utils.coev_budget_sampler import CoEVBudgetedStratifiedSampler
from fate_oia.utils.tida_stateful_sampler import TIDAStatefulRandomSampler


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_model(cfg: dict[str, Any], use_mock_dino: bool = False) -> CoEVOIAModel:
    return CoEVOIAModel(cfg["backbone"]["pretrained_weights"], cfg["model"]["history_chunk_size"],
                        cfg["model"]["reason_soft_bias_verified"], use_mock_dino,
                        cfg["backbone"]["activation_checkpointing"])


def build_optimizer(model: CoEVOIAModel, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    upper_ids = {id(p) for p in model.visual_field.backbone.blocks[8:].parameters()} | {id(p) for p in model.visual_field.task_norm.parameters()}
    groups = {"upper_decay": [], "upper_no_decay": [], "new_decay": [], "new_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        owner = "upper" if id(p) in upper_ids else "new"
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        groups[f"{owner}_{'no_decay' if no_decay else 'decay'}"].append(p)
    listed = [id(p) for group in groups.values() for p in group]
    if len(listed) != len(set(listed)) or set(listed) != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError("optimizer owner exact-cover failed")
    parameter_groups = []
    for name, params in groups.items():
        if params:
            parameter_groups.append({"params": params,
                                     "lr": cfg["training"]["lr_upper_dino"] if name.startswith("upper") else cfg["training"]["lr_new_modules"],
                                     "weight_decay": 0.0 if name.endswith("no_decay") else cfg["training"]["weight_decay"],
                                     "owner": name})
    return torch.optim.AdamW(parameter_groups)


def lr_factor(update: int, total: int, warmup: int, minimum: float) -> float:
    if update < warmup: return (update + 1) / max(1, warmup)
    progress = (update - warmup) / max(1, total - warmup)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * min(1., progress)))


def module_grad_norm(module: torch.nn.Module) -> float:
    values=[p.grad.float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def write_epoch_row(path: Path, row: dict[str,Any]) -> None:
    existing=[]
    if path.is_file():
        existing=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    existing=[value for value in existing if value.get("epoch")!=row.get("epoch")]+[row]
    existing.sort(key=lambda value:value["epoch"]);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text("".join(json.dumps(value)+"\n" for value in existing),encoding="utf-8");temp.replace(path)


def loaders(cfg: dict[str, Any], batch_size: int, max_train: int | None = None, max_test: int | None = None):
    data = cfg["data"]
    history_frames=int(data["history_frames"])
    train = CoEVVideoDataset(data["manifest_path"], data["train_partitions"], True, data["grounding_root"], cfg["training"]["seed"], max_train,history_frames)
    test = CoEVVideoDataset(data["manifest_path"], "test", False, None, cfg["training"]["seed"], max_test,history_frames)
    budget=data.get("epoch_budget",{})
    if max_train is None and budget.get("enabled"):
        sampler = CoEVBudgetedStratifiedSampler(train,seed=cfg["training"]["seed"],metadata_path=budget["novelty_metadata_path"],
            epoch_size=budget["epoch_size"],quotas=budget["quotas"],
            missing_history_quota=budget["missing_history_quota"],candidate_draws=budget["candidate_draws"])
    else:
        sampler = TIDAStatefulRandomSampler(train, seed=cfg["training"]["seed"])
    common = dict(batch_size=batch_size, num_workers=data["num_workers"], pin_memory=data["pin_memory"], collate_fn=coev_collate)
    if data["num_workers"]:
        common.update(persistent_workers=data["persistent_workers"], prefetch_factor=data["prefetch_factor"])
    return DataLoader(train, sampler=sampler, **common), DataLoader(test, shuffle=False, generator=torch.Generator().manual_seed(77), **common), sampler


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--ready-manifest")
    parser.add_argument("--resume"); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--grad-accum", type=int, default=32)
    parser.add_argument("--max-train", type=int); parser.add_argument("--max-test", type=int); parser.add_argument("--epochs", type=int)
    args = parser.parse_args(); cfg = load_config(args.config)
    if args.batch_size * args.grad_accum != cfg["training"]["effective_batch"]:
        raise ValueError("batch_size * grad_accum must preserve effective batch 32")
    if cfg["runtime"]["feature_cache_enabled"] or cfg["runtime"]["token_compression"] != "none": raise RuntimeError("cache/compression forbidden")
    if not args.ready_manifest:
        raise ValueError("formal COEV entry requires --ready-manifest")
    ready_path = Path(args.ready_manifest)
    if not ready_path.is_file(): raise FileNotFoundError(ready_path)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if not ready.get("pass"): raise RuntimeError("FULL_TRAIN_READY is not passing")
    seed = cfg["training"]["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda"); model = build_model(cfg).to(device); loss_fn = CoEVLoss(); optimizer = build_optimizer(model, cfg)
    train_loader, test_loader, sampler = loaders(cfg, args.batch_size, args.max_train, args.max_test)
    total = cfg["training"]["total_updates"]; warmup = cfg["training"]["warmup_updates"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda u: lr_factor(u, total, warmup, cfg["training"]["min_lr_ratio"]))
    output = Path(cfg["runtime"]["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    identity = runtime_identity(args.config, ready_path.parent / "data_audit.json")
    if ready.get("source_commit") != identity["git_head"] or ready.get("config_sha256") != identity["config_sha256"]:
        raise RuntimeError("ready manifest is stale for the current HEAD/config")
    (output/"config_resolved.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    (output/"run_manifest.json").write_text(json.dumps({"config":args.config,"seed":seed,"epochs":args.epochs or cfg["training"]["epochs"],
        "batch_size":args.batch_size,"gradient_accumulation_steps":args.grad_accum,"effective_batch":args.batch_size*args.grad_accum,
        "test_only_eval":True,"threshold":.5,"internal_test_selected":True,"publication_eligible":False,
        "feature_cache_enabled":False,"token_compression":"none","pretrained_weights":cfg["backbone"]["pretrained_weights"],
        "identity":identity,"spec_sha256":cfg["experiment"]["spec_sha256"],"total_updates":cfg["training"]["total_updates"],
        "warmup_updates":cfg["training"]["warmup_updates"],"foreground_only":cfg["runtime"]["foreground_only"],
        "history_frames":cfg["data"]["history_frames"],"temporal_span_seconds":cfg["data"]["temporal_span_seconds"],
        "epoch_budget":cfg["data"]["epoch_budget"]},indent=2),encoding="utf-8")
    epoch0 = update = 0; best = None; best_epoch = None
    if args.resume:
        state = torch.load(args.resume, map_location="cpu"); model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        if not state.get("optimizer_boundary", False): raise RuntimeError("resume checkpoint is not at an optimizer boundary")
        verify_resume_identity(state["identity"], identity)
        if state["total_updates"] != cfg["training"]["total_updates"]: raise RuntimeError("resume total_updates mismatch")
        scheduler.load_state_dict(state["scheduler"]); sampler.load_state_dict(state["sampler"]); restore_rng(state["rng"])
        epoch0, update, best, best_epoch = state["epoch"], state["global_update"], state["best"], state["best_epoch"]
    epochs = args.epochs or cfg["training"]["epochs"]
    scaler_ctx = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    for epoch in range(epoch0, epochs):
        train_loader.dataset.set_epoch(epoch); model.train(); optimizer.zero_grad(set_to_none=True); group = []
        epoch_sampler_stats=sampler.current_stats() if hasattr(sampler,"current_stats") else {"epoch":epoch,"selected_count":len(train_loader.dataset)}
        epoch_target=len(sampler)+sampler.consumed
        last_batch_end=time.perf_counter()
        for micro, (inputs, targets) in enumerate(train_loader):
            load_seconds=time.perf_counter()-last_batch_end
            group.append((inputs, targets)); group_samples = sum(x.action.shape[0] for _, x in group)
            boundary = len(group) == args.grad_accum or sampler.consumed + group_samples == epoch_target
            if not boundary: continue
            before = sampler.consumed
            last_out = None; log_due=(update+1) % cfg["runtime"]["log_every_updates"] == 0
            model.capture_gradient_diagnostics=log_due
            forward_seconds=backward_seconds=0.0
            parameter_before={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad} if log_due else {}
            for ins, tgt in group:
                tick=time.perf_counter()
                with scaler_ctx():
                    last_out = model(ins.to(device)); losses = loss_fn(last_out, tgt.to(device))
                forward_seconds += time.perf_counter()-tick; tick=time.perf_counter()
                (losses["total"] * (tgt.action.shape[0] / group_samples)).backward()
                backward_seconds += time.perf_counter()-tick
            if not torch.isfinite(torch.stack([p.grad.float().norm() for p in model.parameters() if p.grad is not None])).all(): raise FloatingPointError("non-finite gradient")
            owner_grad={}
            history_grad=[]
            cancellation_ratio=0.0
            if log_due:
                owner_grad={"dino9":module_grad_norm(model.visual_field.backbone.blocks[8]),"dino10":module_grad_norm(model.visual_field.backbone.blocks[9]),
                            "dino11":module_grad_norm(model.visual_field.backbone.blocks[10]),"dino12":module_grad_norm(model.visual_field.backbone.blocks[11]),
                            "decoder":module_grad_norm(model.video_decoder),"predicate":module_grad_norm(model.predicate_observer),
                            "matcher":module_grad_norm(model.correspondence_observer),"readout":module_grad_norm(model.evidence_readout)}
                for field in last_out.get("history_gradient_fields", ()):
                    middle=field.shape[1]//2;late=field.shape[1]-1
                    history_grad.append({"early":float(field.grad[:,0].float().norm()) if field.grad is not None else 0.0,
                                         "middle":float(field.grad[:,middle].float().norm()) if field.grad is not None else 0.0,
                                         "late":float(field.grad[:,late].float().norm()) if field.grad is not None else 0.0})
                contribution=last_out["factor_contribution"].float()
                cancellation_ratio=float(contribution.abs().sum(-1).mean()/contribution.sum(-1).abs().mean().clamp_min(1e-8))
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); sampler.mark_consumed(group_samples); update += 1
            update_norm=(sum((p.detach()-parameter_before[name]).float().square().sum() for name,p in model.named_parameters() if name in parameter_before).sqrt().item() if log_due else 0.0)
            if update % cfg["runtime"]["log_every_updates"] == 0:
                assert last_out is not None
                print(json.dumps({"event":"coev_batch","epoch":epoch,"update":update,"consumed":sampler.consumed,
                    "loss":float(losses["total"]),"loss_action":float(losses["action_task"]),"loss_reason":float(losses["reason_task"]),
                    "loss_ground":float(losses["ground"]),"loss_match":float(losses["match"]),"grad_norm":float(grad),
                    "primitive_valid_rate":float(last_out["primitive_valid"].float().mean()),
                    "pair_valid_rate":float(last_out["pair_valid"].float().mean()),
                    "area_abs_mean":float(last_out["pair"][...,11].abs().mean()),"product_mean":float(last_out["pair"][...,10].mean()),
                    "visual_rms":float(last_out["visual_logits"].float().square().mean().sqrt()),
                    "evidence_rms":float(last_out["evidence_logits"].float().square().mean().sqrt()),
                    "matched_mass":float(last_out["matcher_forward"][...,:-1].sum(-1).mean()),
                    "correspondence_quality_rate":float(last_out["correspondence_quality_rate"].mean()),
                    "ground_known_rate":float(targets.grounding_known.float().mean()),
                    "factor_cancellation_ratio":cancellation_ratio,
                    "owner_grad":owner_grad,
                    "history_grad":history_grad,"update_norm":update_norm,"load_seconds":load_seconds,
                    "forward_seconds":forward_seconds,"backward_seconds":backward_seconds,
                    "gpu_name":torch.cuda.get_device_name(device),"gpu_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                    "gpu_reserved_gib":torch.cuda.max_memory_reserved()/2**30}), flush=True)
            if update % cfg["runtime"]["save_every_updates"] == 0:
                atomic_torch_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"sampler":sampler.state_dict(),"rng":capture_rng(),"epoch":epoch,"global_update":update,"best":best,"best_epoch":best_epoch,"optimizer_boundary":True,"identity":identity,"total_updates":total}, output/"checkpoint_latest.pth")
            group = []
            model.capture_gradient_diagnostics=False; last_batch_end=time.perf_counter()
        if sampler.epoch_complete: sampler.advance_epoch()
        if hasattr(sampler, "exposure"):
            epoch_sampler_stats["pool_covered_after_epoch_rate"] = float((sampler.exposure > 0).float().mean())
            epoch_sampler_stats["exposure_min_after_epoch"] = int(sampler.exposure.min())
            epoch_sampler_stats["exposure_max_after_epoch"] = int(sampler.exposure.max())
        metrics, _ = evaluate(model, test_loader, device); metrics.update(epoch=epoch, raw_fixed_0p5=True, internal_test_selected=True)
        write_epoch_row(output/"metrics_summary.jsonl",metrics)
        write_epoch_row(output/"sampler_epoch_stats.jsonl",epoch_sampler_stats)
        traffic=final_intervention_diagnostics(model,test_loader,device,cfg["runtime"]["traffic_audit_samples_per_epoch"])
        traffic["epoch"]=epoch;write_epoch_row(output/"traffic_interventions_test512.jsonl",traffic)
        state = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"sampler":sampler.state_dict(),"rng":capture_rng(),"epoch":epoch+1,"global_update":update,"best":best,"best_epoch":best_epoch,"optimizer_boundary":True,"identity":identity,"total_updates":total}
        if is_better(metrics,best,epoch,best_epoch): best,best_epoch=metrics,epoch; state["best"],state["best_epoch"]=best,best_epoch; atomic_torch_save(state,output/"checkpoint_best_joint.pth")
        atomic_torch_save(state, output/"checkpoint_latest.pth")
    formal = epochs == cfg["training"]["epochs"] and args.max_train is None and args.max_test is None
    if formal:
        best_state=torch.load(output/"checkpoint_best_joint.pth",map_location="cpu");model.load_state_dict(best_state["model"])
        final_limit=len(test_loader.dataset) if cfg["runtime"]["final_traffic_audit_full_test"] else cfg["runtime"]["traffic_audit_samples_per_epoch"]
        diagnostics=final_intervention_diagnostics(model,test_loader,device,final_limit)
        (output/"final_interventions_full_test.json").write_text(json.dumps(diagnostics,indent=2),encoding="utf-8")
        (output/"TRAIN_COMPLETED.json").write_text(json.dumps({"epochs":epochs,"test_evaluations":epochs,"best":best,"best_epoch":best_epoch,"final_diagnostics":True}),encoding="utf-8")


if __name__ == "__main__": main()
