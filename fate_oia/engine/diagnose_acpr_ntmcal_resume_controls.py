from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from fate_oia.engine.train_acpr_ntmcal_oia import (
    build_model,
    load_config,
    make_dataset,
    make_loader,
    make_loader_from_dataset,
    set_lrs,
    update_train_calib_teacher,
)
from fate_oia.engine.eval_acpr_ntmcal_oia import evaluate_ntmcal_tensors
from fate_oia.losses.acpr_ntmcal_losses import (
    action_asl_loss,
    action_predicate_margin_loss,
    native_predicate_measurement_loss,
    ntmcal_calibration_loss,
    ntmcal_reason_pu_loss,
    predicate_attention_sparsity_loss,
    schedule_weights,
)
from fate_oia.models.acpr_ntmcal_text_atoms import native_text_structure_loss
from fate_oia.utils.acpr_ntmcal_artifacts import append_jsonl, save_tensor, write_json
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


BASELINE_EPOCH8_ACT = 0.703643262386322
BASELINE_EPOCH8_EXP = 0.34061235189437866


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def variant_flags(variant: str) -> dict[str, bool]:
    table = {
        "R8-A": {"disable_action_predicate": True, "disable_pair": False, "disable_action_threshold_delta": False, "pcgrad": False},
        "R8-B": {"disable_action_predicate": False, "disable_pair": True, "disable_action_threshold_delta": False, "pcgrad": False},
        "R8-C": {"disable_action_predicate": False, "disable_pair": False, "disable_action_threshold_delta": True, "pcgrad": False},
        "R8-D": {"disable_action_predicate": True, "disable_pair": True, "disable_action_threshold_delta": False, "pcgrad": False},
        "R8-E": {"disable_action_predicate": False, "disable_pair": False, "disable_action_threshold_delta": False, "pcgrad": True},
    }
    if variant not in table:
        raise ValueError(f"unknown variant {variant}; expected one of {sorted(table)}")
    return table[variant]


def recompute_threshold(model, out: dict[str, Any], action_logits: torch.Tensor, reason_logits: torch.Tensor, epoch: int, *, disable_action_threshold_delta: bool) -> dict[str, Any]:
    cal = model.ntmcal_threshold(
        action_logits,
        reason_logits,
        out["support_score"],
        out["contra_score"],
        out["pu_state"]["reason_rho"],
        out["reason_logits_base"],
        out["predicate_q"],
        out["predicate_rho"],
        epoch=epoch,
    )
    if disable_action_threshold_delta:
        theta_action = model.ntmcal_threshold.theta_action_global.view(1, -1)
        cal["threshold_delta_action"] = torch.zeros_like(cal["threshold_delta_action"])
        cal["theta_action"] = theta_action.expand_as(action_logits)
        cal["action_logits_deploy"] = action_logits - theta_action
        cal["logits_deploy"] = torch.cat([cal["action_logits_deploy"], cal["reason_logits_deploy"]], dim=-1)
        cal["threshold_stats"] = {
            **cal["threshold_stats"],
            "threshold_delta_action_abs_mean": 0.0,
            "action_threshold_delta_disabled": True,
        }
    return cal


def apply_variant_controls(model, out: dict[str, Any], epoch: int, flags: dict[str, bool]) -> dict[str, Any]:
    action_logits = out["action_logits_ntmcal"]
    if flags["disable_action_predicate"]:
        action_logits = out["action_logits_base"]
        out["action_predicate_delta"] = torch.zeros_like(out["action_predicate_delta"])
        out["action_predicate_stats"] = {**out.get("action_predicate_stats", {}), "disabled_by_diagnostic": True}
    reason_logits = out["reason_logits_ntmcal"]
    cal = recompute_threshold(model, out, action_logits, reason_logits, epoch, disable_action_threshold_delta=flags["disable_action_threshold_delta"])
    out.update(cal)
    out["action_logits_ntmcal"] = action_logits
    out["reason_logits_ntmcal"] = reason_logits
    out["branch_logits"] = {
        "base_fixed": torch.cat([out["action_logits_base"], out["reason_logits_base"]], -1),
        "deploy_fixed": out["logits_deploy"],
    }
    if flags["disable_pair"]:
        out.pop("_pair_memory", None)
    return out


def compute_components(out: dict[str, Any], action_targets: torch.Tensor, reason_targets: torch.Tensor, epoch: int) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    comps: dict[str, torch.Tensor] = {}
    comps["action"] = action_asl_loss(out["action_logits_deploy"], action_targets)
    comps["reason"] = ntmcal_reason_pu_loss(out["reason_logits_deploy"], reason_targets, out["pu_state"], epoch)
    comps["predicate"] = native_predicate_measurement_loss(out["predicate_q"], out["predicate_rho"], out["native_text_observations"], epoch)
    comps["text"] = native_text_structure_loss(out["_atom_encoder"], out["_predicate_specs"])["native_text_structure_loss"] if "_atom_encoder" in out else out["action_logits_deploy"].sum() * 0.0
    comps["calibration"] = ntmcal_calibration_loss(out, action_targets, reason_targets)
    comps["action_predicate"] = action_predicate_margin_loss(out, action_targets, epoch)
    comps["sparse"] = predicate_attention_sparsity_loss(out["predicate_topk_attention"])
    main = comps["action"] + comps["reason"]
    if "_pair_memory" in out:
        comps["pair"], pair_stats = out["_pair_memory"].loss(out["reason_logits_deploy"], reason_targets, out["pu_state"], epoch, main_loss=main)
    else:
        comps["pair"], pair_stats = main * 0.0, {
            "pair_count_total": 0,
            "tail_pair_count": 0,
            "zero_pair_count": int(reason_targets.shape[1]),
            "memory_positive_coverage": -1,
            "memory_negative_coverage": -1,
            "cap_applied": False,
        }
    return comps, pair_stats


def weighted_total(comps: dict[str, torch.Tensor], epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    w = schedule_weights(epoch)
    total = (
        w["action"] * comps["action"]
        + w["reason"] * comps["reason"]
        + w["pred"] * comps["predicate"]
        + w["text"] * comps["text"]
        + w["cal"] * comps["calibration"]
        + w["act_pred"] * comps["action_predicate"]
        + w["pair"] * comps["pair"]
        + w["sparse"] * comps["sparse"]
    )
    return total, w


def flatten_grads(loss: torch.Tensor, params: list[torch.nn.Parameter], *, retain_graph: bool) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    flat = []
    for p, g in zip(params, grads):
        if g is None:
            flat.append(torch.zeros_like(p).reshape(-1))
        else:
            flat.append(g.detach().reshape(-1))
    if not flat:
        return torch.zeros(1, device=loss.device)
    return torch.cat(flat)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = a.norm() * b.norm()
    if float(denom.detach().cpu()) <= 0:
        return 0.0
    return float((a @ b / denom).detach().cpu())


def grad_cosine_report(comps: dict[str, torch.Tensor], shared_params: list[torch.nn.Parameter]) -> dict[str, float]:
    ga = flatten_grads(comps["action"], shared_params, retain_graph=True)
    return {
        "cos_action_reason": cosine(ga, flatten_grads(comps["reason"], shared_params, retain_graph=True)),
        "cos_action_predicate": cosine(ga, flatten_grads(comps["predicate"], shared_params, retain_graph=True)),
        "cos_action_calibration": cosine(ga, flatten_grads(comps["calibration"], shared_params, retain_graph=True)),
        "cos_action_pair": cosine(ga, flatten_grads(comps["pair"], shared_params, retain_graph=True)),
        "grad_norm_action": float(ga.norm().detach().cpu()),
    }


def pcgrad_backward(total: torch.Tensor, comps: dict[str, torch.Tensor], weights: dict[str, float], shared_params: list[torch.nn.Parameter], scale: float) -> dict[str, float]:
    action_obj = weights["action"] * comps["action"] * scale
    aux = (
        weights["reason"] * comps["reason"]
        + weights["pred"] * comps["predicate"]
        + weights["cal"] * comps["calibration"]
        + weights["pair"] * comps["pair"]
    ) * scale
    ga = torch.autograd.grad(action_obj, shared_params, retain_graph=True, allow_unused=True)
    gaux = torch.autograd.grad(aux, shared_params, retain_graph=True, allow_unused=True)
    total.backward()
    projected = 0
    for p, a, u in zip(shared_params, ga, gaux):
        if p.grad is None or a is None or u is None:
            continue
        dot = torch.sum(a.detach() * u.detach())
        if dot < 0:
            denom = torch.sum(a.detach() * a.detach()).clamp_min(1e-12)
            new_aux = u.detach() - dot / denom * a.detach()
            p.grad = a.detach() + new_aux
            projected += 1
    return {"pcgrad_projected_param_count": projected}


@torch.no_grad()
def evaluate(model, loader, device, out_dir: Path, epoch: int, variant: str) -> dict[str, Any]:
    model.eval()
    action_base, reason_base, action_dep, reason_dep, labels_a, labels_r, names = [], [], [], [], [], [], []
    flags = variant_flags(variant)
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, epoch=epoch, split="test", reason_labels=None, file_names=batch["file_name"], structured_records=None)
        out = apply_variant_controls(model, out, epoch, flags)
        action_base.append(out["action_logits_base"].detach().cpu())
        reason_base.append(out["reason_logits_base"].detach().cpu())
        action_dep.append(out["action_logits_deploy"].detach().cpu())
        reason_dep.append(out["reason_logits_deploy"].detach().cpu())
        labels_a.append(batch["action"])
        labels_r.append(batch["reason"])
        names.extend(batch["file_name"])
    tensors = {
        "action_base": torch.cat(action_base),
        "reason_base": torch.cat(reason_base),
        "action_deploy": torch.cat(action_dep),
        "reason_deploy": torch.cat(reason_dep),
        "labels_action": torch.cat(labels_a),
        "labels_reason": torch.cat(labels_r),
    }
    metrics = evaluate_ntmcal_tensors(tensors["action_base"], tensors["reason_base"], tensors["action_deploy"], tensors["reason_deploy"], tensors["labels_action"], tensors["labels_reason"])
    metrics["epoch"] = epoch
    metrics["variant"] = variant
    for key, tensor in tensors.items():
        save_tensor(out_dir / f"{key}_test.pt", tensor)
    write_json(out_dir / "file_names_test.json", names)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--variant", required=True, choices=["R8-A", "R8-B", "R8-C", "R8-D", "R8-E"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=200)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flags = variant_flags(args.variant)
    train_dataset = make_dataset(cfg, "train")
    main_idx, calib_idx = make_train_calib_indices(train_dataset, calib_fraction=0.10)
    if args.max_train_samples:
        main_idx = main_idx[: args.max_train_samples]
        calib_idx = calib_idx[: max(1, min(len(calib_idx), args.max_train_samples))]
    train_loader = make_loader_from_dataset(train_dataset, cfg, args.batch_size, None, True, args.num_workers, main_idx)
    train_calib_loader = make_loader_from_dataset(train_dataset, cfg, args.batch_size, None, False, args.num_workers, calib_idx)
    test_loader = make_loader(cfg, "test", args.batch_size, args.max_test_samples, False, args.num_workers)
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    start_epoch = int(ckpt.get("epoch", 8)) + 1
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg.get("training", {}).get("lr_trunk", 2e-4)), weight_decay=0.05)
    shared_params = [p for p in model.trunk.parameters() if p.requires_grad]
    shared_params.extend([p for p in model.predicate_measurement.parameters() if p.requires_grad])
    write_json(out_dir / "run_manifest.json", {
        "variant": args.variant,
        "flags": flags,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "purpose": "D2 short resume + D3 gradient conflict audit",
    })
    below_count = 0
    stop_reason = None
    for local_epoch in range(args.epochs):
        epoch = start_epoch + local_epoch
        model.train()
        set_lrs(opt, min(epoch, int(cfg.get("training", {}).get("epochs", 18)) - 1), cfg)
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                out = model(images, epoch=epoch, split="train", reason_labels=reason, file_names=batch["file_name"], structured_records=None)
                out["_atom_encoder"] = model.atom_encoder
                out["_predicate_specs"] = model.predicate_bank.specs
                out["_pair_memory"] = model.pair_memory
                out = apply_variant_controls(model, out, epoch, flags)
                comps, pair_stats = compute_components(out, action, reason, epoch)
                total, weights = weighted_total(comps, epoch)
                loss = total / args.gradient_accumulation_steps
            need_log = step == 1 or step % args.log_every == 0
            grad_report = grad_cosine_report(comps, shared_params) if need_log else {}
            extra = {}
            if flags["pcgrad"]:
                extra = pcgrad_backward(loss, comps, weights, shared_params, 1.0 / args.gradient_accumulation_steps)
            else:
                loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
            if need_log:
                row = {
                    "variant": args.variant,
                    "epoch": epoch,
                    "local_epoch": local_epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "lr": opt.param_groups[0]["lr"],
                    "loss_total": float(total.detach().cpu()),
                    "loss_action": float(comps["action"].detach().cpu()),
                    "loss_reason": float(comps["reason"].detach().cpu()),
                    "loss_predicate": float(comps["predicate"].detach().cpu()),
                    "loss_calibration": float(comps["calibration"].detach().cpu()),
                    "loss_pair": float(comps["pair"].detach().cpu()),
                    **{f"weight_{k}": v for k, v in weights.items()},
                    **pair_stats,
                    **grad_report,
                    **extra,
                }
                append_jsonl(out_dir / "loss_components.jsonl", row)
                append_jsonl(out_dir / "grad_cosine.jsonl", row)
                print("ntmcal_d2_batch " + json.dumps(row, ensure_ascii=False), flush=True)
        teacher = update_train_calib_teacher(model, train_calib_loader, device, epoch)
        metrics = evaluate(model, test_loader, device, out_dir, epoch, args.variant)
        metrics["teacher"] = teacher
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "variant": args.variant}, out_dir / "checkpoint_latest.pth")
        act = float(metrics["metrics_deploy_fixed"]["Act_mF1"])
        exp = float(metrics["metrics_deploy_fixed"]["Exp_mF1"])
        if act < BASELINE_EPOCH8_ACT - 0.005:
            below_count += 1
        else:
            below_count = 0
        if below_count >= 2:
            stop_reason = f"Act_mF1 below epoch8-0.005 for two consecutive epochs: act={act:.6f}"
        if exp > BASELINE_EPOCH8_EXP and act < BASELINE_EPOCH8_ACT - 0.008:
            stop_reason = f"explanation-over-regularization: exp={exp:.6f} > baseline {BASELINE_EPOCH8_EXP:.6f}, act={act:.6f} < baseline-0.008"
        print("ntmcal_d2_epoch " + json.dumps({"variant": args.variant, "epoch": epoch, "Act_mF1": act, "Act_oF1": metrics["metrics_deploy_fixed"]["Act_oF1"], "Exp_mF1": exp, "Exp_oF1": metrics["metrics_deploy_fixed"]["Exp_oF1"], "stop_reason": stop_reason}, ensure_ascii=False), flush=True)
        write_json(out_dir / "live_status.json", {"variant": args.variant, "epoch": epoch, "Act_mF1": act, "Exp_mF1": exp, "stop_reason": stop_reason})
        if stop_reason:
            break
    write_json(out_dir / "diagnostic_completed.json", {"variant": args.variant, "stop_reason": stop_reason or "completed_requested_epochs"})


if __name__ == "__main__":
    main()
