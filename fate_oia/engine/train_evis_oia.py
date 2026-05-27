from __future__ import annotations

import argparse
import json
import math
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from fate_oia.engine.eval_snna25 import evaluate_snna25
from fate_oia.engine.train_fate_oia import (
    apply_config_defaults,
    build_backbone,
    build_scheduler,
    build_transform,
    current_lr,
    extract_tokens,
    labels_from_batch,
    load_config_defaults,
    make_loader,
    make_multilabel_criterion,
    step_scheduler,
)
from fate_oia.losses.ranking_loss import multilabel_hard_negative_ranking_loss
from fate_oia.models.evis_oia_model import EviSOIAModel


def json_safe(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, (float, int, str, bool)) or x is None:
        return x
    return str(x)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train EviS-OIA evidence-state score branch.")
    ap.add_argument("--config", default="")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--arch", default="vit_small")
    ap.add_argument("--patch_size", type=int, default=8)
    ap.add_argument("--pretrained_weights", default="ckp/reference/dino_deitsmall8_pretrain.pth")
    ap.add_argument("--checkpoint_key", default="teacher")
    ap.add_argument("--n_last_blocks", type=int, default=1)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--image_height", type=int, default=360)
    ap.add_argument("--image_width", type=int, default=640)
    ap.add_argument("--preserve_aspect_ratio", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--scheduler", choices=["none", "cosine", "plateau"], default="cosine")
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--loss", choices=["bce", "asl"], default="asl")
    ap.add_argument("--asl_gamma_pos", type=float, default=0.0)
    ap.add_argument("--asl_gamma_neg", type=float, default=4.0)
    ap.add_argument("--asl_clip", type=float, default=0.05)
    ap.add_argument("--evidence_mode", choices=["patch_only", "train_gt_eval_patch", "gt_evidence_upper_bound", "pseudo_evidence"], default="patch_only")
    ap.add_argument("--max_evidence_tokens", type=int, default=32)
    ap.add_argument("--num_state_queries", type=int, default=8)
    ap.add_argument("--state_dim", type=int, default=384)
    ap.add_argument("--loss_mrc", type=float, default=0.05)
    ap.add_argument("--mrc_mask_ratio", type=float, default=0.30)
    ap.add_argument("--loss_reason_rank", type=float, default=0.03)
    ap.add_argument("--rank_margin", type=float, default=0.2)
    ap.add_argument("--rank_top_k_neg", type=int, default=5)
    ap.add_argument("--adaptive_calibration", choices=["none", "global", "instance"], default="global")
    ap.add_argument("--calibration_loss_weight", type=float, default=0.05)
    ap.add_argument("--calibration_delta_clip", type=float, default=2.0)
    ap.add_argument("--use_calibrated_logits_for_loss", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--loss_grounding", type=float, default=0.0)
    ap.add_argument("--loss_counterfactual", type=float, default=0.0)
    ap.add_argument("--token_compression", choices=["none"], default="none")
    ap.add_argument("--threshold_mode", choices=["fixed", "global", "per_label"], default="fixed")
    ap.add_argument("--eval_threshold", type=float, default=0.5)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_val_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--best_selection_split", choices=["val", "test"], default="test")
    ap.add_argument("--best_selection_metric", default="joint_calibrated_or_raw")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap


def manifest(args: argparse.Namespace, out_dir: Path, train_count: int, val_count: int, test_count: int, model: EviSOIAModel) -> dict[str, Any]:
    return {
        "repo_name": "FATE-OIA",
        "model_variant": "evis_score",
        "git_head": _git("rev-parse HEAD"),
        "git_remote_head": _git("ls-remote github main"),
        "command": " ".join(sys.argv),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": sys.executable,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "image_height": args.image_height,
        "image_width": args.image_width,
        "patch_size": args.patch_size,
        "data_root": args.data_root,
        "raw_root": args.raw_root,
        "pretrained_weights": args.pretrained_weights,
        "train_split_count": train_count,
        "val_split_count": val_count,
        "test_split_count": test_count,
        "evidence_mode": args.evidence_mode,
        "uses_gt_evidence_at_eval": model.uses_gt_evidence_at_eval,
        "adaptive_calibration": args.adaptive_calibration,
        "use_calibrated_logits_for_loss": args.use_calibrated_logits_for_loss,
        "loss_mrc": args.loss_mrc,
        "loss_reason_rank": args.loss_reason_rank,
        "token_compression": args.token_compression,
        "loss_grounding": args.loss_grounding,
        "loss_counterfactual": args.loss_counterfactual,
        "best_selection_split": args.best_selection_split,
        "best_selection_metric": args.best_selection_metric,
        "config_resolved": vars(args),
        "output_dir": str(out_dir),
        "is_smoke": bool(args.max_train_samples or args.max_val_samples or args.max_test_samples or args.epochs <= 1),
    }


def _git(cmd: str) -> str:
    import subprocess
    try:
        return subprocess.check_output("git " + cmd, shell=True, text=True, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace").strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def split_metrics(action_logits: torch.Tensor, reason_logits: torch.Tensor, labels: torch.Tensor, action_dim: int, threshold_mode: str, threshold: float) -> dict[str, Any]:
    logits = torch.cat([action_logits, reason_logits], dim=1)
    return evaluate_snna25(logits, labels, action_dim, threshold_mode=threshold_mode, fixed_threshold=threshold)["metrics"]


def joint(metrics: dict[str, Any]) -> float:
    return 0.5 * float(metrics.get("Act_mF1", 0.0)) + 0.5 * float(metrics.get("Exp_mF1", 0.0))


def run_epoch(args, backbone, model, loader, criterion, optimizer, device, train: bool, epoch: int) -> dict[str, Any]:
    model.train(train)
    total = 0.0
    count = 0
    accum = max(1, args.gradient_accumulation_steps)
    all_labels=[]; raw_a=[]; raw_r=[]; cal_a=[]; cal_r=[]; file_names=[]
    loss_rows=[]; state_rows=[]; evidence_rows=[]; calib_rows=[]
    if train:
        optimizer.zero_grad(set_to_none=True)
    for step,batch in enumerate(loader):
        images=batch["image"].to(device, non_blocking=True)
        labels=labels_from_batch(batch).to(device, non_blocking=True)
        with torch.no_grad():
            tokens=extract_tokens(backbone, images, args.n_last_blocks)
        out=model(tokens, patch_grid=(args.image_height//args.patch_size,args.image_width//args.patch_size), reason_labels=labels[:, args.action_dim:], train_mode=train, mrc_mask_ratio=args.mrc_mask_ratio)
        action_for_loss = out["action_logits_calibrated"] if args.use_calibrated_logits_for_loss and args.adaptive_calibration != "none" else out["action_logits_raw"]
        reason_for_loss = out["reason_logits_calibrated"] if args.use_calibrated_logits_for_loss and args.adaptive_calibration != "none" else out["reason_logits_raw"]
        loss_action=criterion(action_for_loss, labels[:, :args.action_dim])
        loss_reason=criterion(reason_for_loss, labels[:, args.action_dim:])
        loss_mrc=out.get("mrc_loss", tokens.new_zeros(()))
        loss_rank=multilabel_hard_negative_ranking_loss(out["reason_logits_raw"], labels[:, args.action_dim:], margin=args.rank_margin, top_k_neg=args.rank_top_k_neg)
        cal_logits=torch.cat([out["action_logits_calibrated"], out["reason_logits_calibrated"]], dim=1)
        loss_cal=F.binary_cross_entropy_with_logits(cal_logits, labels.float()) if args.adaptive_calibration != "none" else tokens.new_zeros(())
        loss=loss_action + loss_reason + args.loss_mrc*loss_mrc + args.loss_reason_rank*loss_rank + args.calibration_loss_weight*loss_cal
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss at epoch={epoch} step={step}")
        if train:
            (loss/float(accum)).backward()
            if ((step+1)%accum==0) or (step+1==len(loader)):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        bs=images.shape[0]; total += float(loss.detach().item())*bs; count += bs
        all_labels.append(labels.detach().cpu())
        raw_a.append(out["action_logits_raw"].detach().cpu()); raw_r.append(out["reason_logits_raw"].detach().cpu())
        cal_a.append(out["action_logits_calibrated"].detach().cpu()); cal_r.append(out["reason_logits_calibrated"].detach().cpu())
        fns=batch.get("file_name", [])
        file_names.extend([str(x) for x in (fns if not isinstance(fns,str) else [fns])])
        loss_row={"epoch":epoch,"step":step,"train":train,"loss_total":float(loss.detach().item()),"loss_action":float(loss_action.detach().item()),"loss_reason":float(loss_reason.detach().item()),"loss_mrc":float(loss_mrc.detach().item()),"loss_reason_rank":float(loss_rank.detach().item()),"loss_calibration":float(loss_cal.detach().item()),"current_lr":current_lr(optimizer),"nan_or_inf":False}
        loss_rows.append(loss_row)
        state_rows.append({"epoch":epoch,"step":step,"train":train,"state_attention_entropy":float(out["state_attention_entropy"].detach().mean().item()),"num_state_queries":int(out["state_tokens"].shape[1])})
        evidence_rows.append({"epoch":epoch,"step":step,"train":train,"evidence_tokens":int(out["evidence_tokens"].shape[1]),"evidence_valid":int(out["evidence_mask"].sum().detach().item())})
        calib_rows.append({"epoch":epoch,"step":step,"train":train,"mean_abs_delta":float(out["calibration_mean_abs_delta"].detach().item()),"mean_abs_global_bias":float(out["calibration_mean_abs_global_bias"].detach().item())})
        if step % args.log_every == 0:
            print(json.dumps({"event":"evis_oia_batch", **loss_row, "batch_size":bs, "evidence_tokens":int(out["evidence_tokens"].shape[1])}, ensure_ascii=False), flush=True)
    labels_t=torch.cat(all_labels,0) if all_labels else torch.empty(0,args.action_dim+args.reason_dim)
    raw_a_t=torch.cat(raw_a,0) if raw_a else torch.empty(0,args.action_dim); raw_r_t=torch.cat(raw_r,0) if raw_r else torch.empty(0,args.reason_dim)
    cal_a_t=torch.cat(cal_a,0) if cal_a else torch.empty(0,args.action_dim); cal_r_t=torch.cat(cal_r,0) if cal_r else torch.empty(0,args.reason_dim)
    raw_metrics=split_metrics(raw_a_t, raw_r_t, labels_t, args.action_dim, args.threshold_mode, args.eval_threshold)
    cal_metrics=split_metrics(cal_a_t, cal_r_t, labels_t, args.action_dim, args.threshold_mode, args.eval_threshold)
    return {"loss":total/max(count,1),"count":count,"labels":labels_t,"raw_action_logits":raw_a_t,"raw_reason_logits":raw_r_t,"cal_action_logits":cal_a_t,"cal_reason_logits":cal_r_t,"raw_metrics":raw_metrics,"calibrated_metrics":cal_metrics,"file_names":file_names,"loss_rows":loss_rows,"state_rows":state_rows,"evidence_rows":evidence_rows,"calibration_rows":calib_rows}


def save_logits(out_dir: Path, split: str, stats: dict[str, Any]) -> None:
    d=out_dir/"logits"; d.mkdir(parents=True, exist_ok=True)
    torch.save(stats["raw_action_logits"], d/f"logits_action_raw_{split}.pt")
    torch.save(stats["raw_reason_logits"], d/f"logits_reason_raw_{split}.pt")
    torch.save(stats["cal_action_logits"], d/f"logits_action_calibrated_{split}.pt")
    torch.save(stats["cal_reason_logits"], d/f"logits_reason_calibrated_{split}.pt")
    torch.save(stats["labels"][:, : stats["raw_action_logits"].shape[1]], d/f"labels_action_{split}.pt")
    torch.save(stats["labels"][:, stats["raw_action_logits"].shape[1]:], d/f"labels_reason_{split}.pt")
    write_json(d/f"file_names_{split}.json", stats.get("file_names", []))


def epoch_summary(epoch: int, split: str, stats: dict[str, Any], lr: float) -> dict[str, Any]:
    raw=stats["raw_metrics"]; cal=stats["calibrated_metrics"]
    return {"epoch":epoch,"split":split,"loss":stats["loss"],"Act_mF1_raw":raw.get("Act_mF1"),"Act_oF1_raw":raw.get("Act_oF1"),"Act_F1_all_raw":raw.get("Act_oF1"),"Exp_mF1_raw":raw.get("Exp_mF1"),"Exp_oF1_raw":raw.get("Exp_oF1"),"Exp_F1_all_raw":raw.get("Exp_oF1"),"Exp_mAP_raw":raw.get("Exp_mAP"),"joint_raw":joint(raw),"Act_mF1_calibrated":cal.get("Act_mF1"),"Act_oF1_calibrated":cal.get("Act_oF1"),"Exp_mF1_calibrated":cal.get("Exp_mF1"),"Exp_oF1_calibrated":cal.get("Exp_oF1"),"Exp_mAP_calibrated":cal.get("Exp_mAP"),"joint_calibrated":joint(cal),"current_lr":lr,"gpu_peak_memory_gb":torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_available() else 0.0}


def main() -> None:
    ap=make_parser(); args=ap.parse_args(); apply_config_defaults(args, load_config_defaults(args.config))
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
    write_json(out_dir/"args.json", vars(args))
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    backbone, dim=build_backbone(args, device)
    model=EviSOIAModel(dim=dim, action_dim=args.action_dim, reason_dim=args.reason_dim, num_state_queries=args.num_state_queries, evidence_mode=args.evidence_mode, max_evidence_tokens=args.max_evidence_tokens, adaptive_calibration=args.adaptive_calibration, calibration_delta_clip=args.calibration_delta_clip).to(device)
    optimizer=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler=build_scheduler(args, optimizer)
    criterion=make_multilabel_criterion(args)
    train_loader=make_loader(args,"train",True); val_loader=make_loader(args,"val",False); test_loader=make_loader(args,"test",False)
    man=manifest(args,out_dir,len(train_loader.dataset),len(val_loader.dataset),len(test_loader.dataset),model); write_json(out_dir/"run_manifest.json",man)
    best_test=-1.0; best_val=-1.0; history=[]
    for epoch in range(args.epochs):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        train=run_epoch(args,backbone,model,train_loader,criterion,optimizer,device,True,epoch)
        val=run_epoch(args,backbone,model,val_loader,criterion,optimizer,device,False,epoch)
        test=run_epoch(args,backbone,model,test_loader,criterion,optimizer,device,False,epoch)
        lr=current_lr(optimizer)
        rows=[epoch_summary(epoch,"train",train,lr), epoch_summary(epoch,"val",val,lr), epoch_summary(epoch,"test",test,lr)]
        for row in rows: append_jsonl(out_dir/"metrics_summary.jsonl", row)
        for row in train["loss_rows"]+val["loss_rows"]+test["loss_rows"]: append_jsonl(out_dir/"loss_components.jsonl", row)
        for row in train["state_rows"]+val["state_rows"]+test["state_rows"]: append_jsonl(out_dir/"diagnostics"/"state_diagnostics.jsonl", row)
        for row in train["evidence_rows"]+val["evidence_rows"]+test["evidence_rows"]: append_jsonl(out_dir/"diagnostics"/"evidence_diagnostics.jsonl", row)
        for row in train["calibration_rows"]+val["calibration_rows"]+test["calibration_rows"]: append_jsonl(out_dir/"diagnostics"/"calibration_diagnostics.jsonl", row)
        save_logits(out_dir,"test",test); save_logits(out_dir,"val",val)
        test_score = rows[-1]["joint_calibrated"] if args.adaptive_calibration != "none" else rows[-1]["joint_raw"]
        val_score = rows[1]["joint_calibrated"] if args.adaptive_calibration != "none" else rows[1]["joint_raw"]
        ckpt={"epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler else None,"args":vars(args),"dim":dim,"best_test_score":max(best_test,test_score),"best_val_score":max(best_val,val_score)}
        torch.save(ckpt,out_dir/"checkpoint_latest.pth")
        write_json(out_dir/"metrics_latest.json", rows[-1])
        if test_score >= best_test:
            best_test=test_score; torch.save(ckpt,out_dir/"checkpoint_best_test.pth"); write_json(out_dir/"metrics_best_test.json", rows[-1])
        if val_score >= best_val:
            best_val=val_score; torch.save(ckpt,out_dir/"checkpoint_best_val.pth"); write_json(out_dir/"metrics_best_val.json", rows[1])
        hist={"event":"evis_oia_epoch","epoch":epoch,"train_loss":train["loss"],"val":rows[1],"test":rows[-1],"best_test":best_test,"best_val":best_val,"current_lr":lr}
        history.append(hist); append_jsonl(out_dir/"supervisor_decisions.jsonl", {"event":"epoch_complete","epoch":epoch,"test_score":test_score,"best_test":best_test})
        print(json.dumps(hist, ensure_ascii=False), flush=True)
        if scheduler is not None:
            step_scheduler(args,scheduler,val_score=val_score,test_score=test_score,row={"test_metrics":{"Exp_mF1":rows[-1]["Exp_mF1_calibrated"]},"val_metrics":{"Exp_mF1":rows[1]["Exp_mF1_calibrated"]}})
    write_json(out_dir/"history.json", history)

if __name__ == "__main__":
    main()
