from __future__ import annotations

import argparse
import math
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_cert_structured_evidence import AIECertStructuredEvidenceBuilder
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.losses.aie_cert_constraints import AIECertDualState
from fate_oia.losses.aie_cert_loss_registry import AIECertLossRegistry, exact_owner_parameter_groups
from fate_oia.losses.aie_cert_losses import (atom_overlap_ceiling_loss, ecpo_loss, evidence_constraints,
    naming_preference_loss, partial_asl, soft_f1)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.models.aie_cert_oia_model import AIECertOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.aie_cert_artifacts import append_jsonl, component_diagnosis, validate_epoch, write_json
from fate_oia.utils.aie_cert_calibration import AIECertCalibrationGuard
from fate_oia.utils.aie_cert_counterfactual import AIECertCounterfactualEngine, target_signed_margin
from fate_oia.utils.aie_cert_metrics import branch_metrics, evidence_diagnostics
from fate_oia.utils.aie_cert_preference_queue import AIECertPreferenceQueue, PreferenceBatch
from fate_oia.utils.aie_cert_schedule import schedule_values
from fate_oia.utils.aie_calibration import fit_posthoc_thresholds, apply_posthoc_threshold


def load_config(path): return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def collate(rows):
    return {"image": torch.stack([x["image"] for x in rows]), "action": torch.stack([x["action"] for x in rows]),
            "reason": torch.stack([x["reason"] for x in rows]), "file_name": [x["file_name"] for x in rows]}


def make_dataset(cfg, split):
    d = cfg["data"]
    transform = AspectRatioLetterboxTransform(d["image_height"], d["image_width"], patch_size=d["patch_size"])
    return BDDOIAMultiTaskDataset(d["data_root"], d["raw_root"], split=split, action_dim=4, reason_dim=21,
                                   load_image=True, transform=transform)


def make_loader(dataset, batch, shuffle, workers, cfg):
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                  pin_memory=bool(cfg["data"]["pin_memory"]),
                  persistent_workers=bool(cfg["data"]["persistent_workers"]) and workers > 0)
    if workers: kwargs["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def accumulation_divisor(micro_step: int, total_micro_steps: int, accumulation: int) -> int:
    remainder = total_micro_steps % accumulation
    if remainder and micro_step >= total_micro_steps - remainder:
        return remainder
    return accumulation


def build_model(cfg, device, mock=False):
    b, p = cfg["backbone"], cfg["primary"]
    return AIECertOIAModel(dim=p["dim"], selected_layers=tuple(b["selected_layers"]),
        pretrained_weights=b["pretrained_weights"], scene_config=p["scene_predicates"],
        grammar_path=p["reason_grammar"], use_mock_dino=mock, mock_dim=p["dim"]).to(device)


def build_optimizer(model, cfg):
    groups = exact_owner_parameter_groups(model)
    keys = {"primary_core": "lr_primary_core", "predicate_visual": "lr_predicate_visual",
            "action_evidence": "lr_action_evidence", "action_contribution": "lr_action_contribution",
            "reason_private": "lr_reason_private", "naming_readout": "lr_naming"}
    params = [{"params": values, "lr": cfg["training"][keys[name]], "base_lr": cfg["training"][keys[name]], "name": name}
              for name, values in groups.items()]
    return torch.optim.AdamW(params, weight_decay=cfg["training"]["weight_decay"],
                             fused=bool(cfg["training"].get("fused_adamw_if_available", False) and torch.cuda.is_available()))


def owner_gradient_stats(model):
    result = {}
    for owner, parameters in exact_owner_parameter_groups(model).items():
        grads = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
        square = sum(float(grad.square().sum()) for grad in grads)
        total = sum(grad.numel() for grad in grads)
        zeros = sum(int((grad == 0).sum()) for grad in grads)
        result[owner] = {"grad_norm": math.sqrt(square), "zero_grad_rate": zeros / max(total, 1),
                         "parameters_with_grad": len(grads), "parameters_total": len(parameters)}
    dino_grads = [parameter.grad.detach().abs().amax() for parameter in model.foundation.dino.parameters()
                  if parameter.grad is not None]
    result["dino_grad_max"] = float(torch.stack(dino_grads).amax()) if dino_grads else 0.0
    return result


def snapshot_owner_parameters(model):
    return {owner: [parameter.detach().clone() for parameter in parameters]
            for owner, parameters in exact_owner_parameter_groups(model).items()}


def owner_update_rms(model, before):
    result = {}
    for owner, parameters in exact_owner_parameter_groups(model).items():
        square = count = 0
        for previous, current in zip(before[owner], parameters):
            delta = current.detach().float() - previous.float()
            square += float(delta.square().sum()); count += delta.numel()
        result[owner] = math.sqrt(square / max(count, 1))
    return result


def _modified_field(field, mask):
    result = dict(field)
    raw = field["patch_tokens_by_layer"]
    weight = mask[:, None, :, None]
    keep = 1.0 - weight
    background = (raw * keep).sum(2, keepdim=True) / keep.sum(2, keepdim=True).clamp_min(1.0)
    result["patch_tokens_by_layer"] = raw * keep + background * weight
    return result


def run_counterfactual(model, field, output, action_target, schedule):
    b = action_target.shape[0]
    signed = (2.0 * action_target - 1.0)[:, :, None] * output["bounded_contribution"]
    atom_by_action = signed.argmax(-1)
    action_id = signed.amax(-1).argmax(-1)
    rows = torch.arange(b, device=action_target.device)
    atom_id = atom_by_action[rows, action_id]
    selected_map = output["atom_map"][rows, action_id, atom_id]
    region = output["atom_region_mask"][rows, action_id, atom_id] > 0.1
    k = min(64, selected_map.shape[-1])
    def topk_mask(score, own_region):
        masked = score.masked_fill(~own_region, torch.finfo(score.dtype).min)
        return torch.zeros_like(score).scatter(-1, masked.topk(k, -1).indices, 1.0)
    selected = topk_mask(selected_map, region)
    controls = []
    for shift in (97, 211):
        controls.append(topk_mask(selected_map.roll(shift, -1), region & (selected < .5)))
    wrong_probe_id = (atom_id + 1) % output["atom_map"].shape[2]
    wrong_probe = topk_mask(output["atom_map"][rows, action_id, wrong_probe_id],
                            (output["atom_region_mask"][rows, action_id, wrong_probe_id] > .1) & (selected < .5))
    wrong_action_id = (action_id + 1) % 4
    wrong_action = topk_mask(output["atom_map"][rows, wrong_action_id, atom_id],
                             (output["atom_region_mask"][rows, wrong_action_id, atom_id] > .1) & (selected < .5))
    controls.extend([wrong_probe, wrong_action])
    original = target_signed_margin(output["action_logits_final"], action_target)[rows, action_id]
    selected_out = model.decode_from_field(_modified_field(field, selected), action_scale=schedule["action_scale"],
        reason_budget_max=schedule["reason_budget_max"], predicate_prior_scale=schedule["predicate_prior_scale"],
        transport_gamma_cap=schedule["transport_gamma_cap"])
    selected_margin = target_signed_margin(selected_out["action_logits_final"], action_target)[rows, action_id]
    control_margins = []
    valid = []
    for control in controls:
        control_out = model.decode_from_field(_modified_field(field, control), action_scale=schedule["action_scale"],
            reason_budget_max=schedule["reason_budget_max"], predicate_prior_scale=schedule["predicate_prior_scale"],
            transport_gamma_cap=schedule["transport_gamma_cap"])
        control_margins.append(target_signed_margin(control_out["action_logits_final"], action_target)[rows, action_id])
        overlap = (control * selected).sum(-1) / selected.sum(-1).clamp_min(1e-8)
        valid.append((control.sum(-1) > 0) & (overlap <= 0.20))
    cf = AIECertCounterfactualEngine().summarize(original, selected_margin, torch.stack(control_margins, -1),
                                                 torch.stack(valid, -1))
    cf.update(action_id=action_id, atom_id=atom_id, selected_mask=selected)
    return cf


def build_ecpo(output, batch, structured, update, queue, threshold=0.50, cap=8):
    losses, gains, weights, pairs = [], [], [], 0
    pair_labels = torch.zeros(21, dtype=torch.long)
    reason = batch["reason"]
    for label in range(21):
        positives = torch.where(reason[:, label] > 0.5)[0][:cap]
        negatives = torch.where((reason[:, label] < 0.5) & (structured["reason_verified_counter"][:, label] > 0.5)
                                & (structured["reason_counter_reliability"][:, label] >= threshold)
                                & (structured["reason_observable_mask"][:, label] > 0.5))[0][:cap]
        if positives.numel() and negatives.numel():
            count = min(positives.numel(), negatives.numel())
            p, n = positives[:count], negatives[:count]
            weight = structured["reason_counter_reliability"][n, label]
            loss = ecpo_loss(output["reason_logits_final_train"][p, label], output["reason_logits_final_train"][n, label],
                output["reason_logits_primary"][p, label], output["reason_logits_primary"][n, label], weight)
            losses.append(loss)
            gain = ((output["reason_logits_final_train"][p, label] - output["reason_logits_final_train"][n, label]) -
                    (output["reason_logits_primary"][p, label] - output["reason_logits_primary"][n, label]).detach())
            gains.append(gain); weights.append(weight); pairs += count; pair_labels[label] += int(count)
        eligible = queue.eligible(update)
        queued_pos = [row for row in eligible if row["reason_target"][label] > .5][:cap]
        queued_neg = [row for row in eligible if row["reason_target"][label] < .5 and
                      row["verified_counter"][label] > .5 and row["counter_reliability"][label] >= threshold][:cap]
        if positives.numel() and queued_neg:
            count = min(positives.numel(), len(queued_neg)); p = positives[:count]
            q_final = torch.stack([row["final_reason_logits"][label] for row in queued_neg[:count]]).to(output["reason_delta"])
            q_primary = torch.stack([row["primary_reason_logits"][label] for row in queued_neg[:count]]).to(output["reason_delta"])
            age = output["reason_delta"].new_tensor([update-row["enqueue_update"] for row in queued_neg[:count]])
            reliability = output["reason_delta"].new_tensor([row["counter_reliability"][label] for row in queued_neg[:count]])
            weight = reliability * torch.exp(-age / queue.age_tau)
            losses.append(ecpo_loss(output["reason_logits_final_train"][p,label], q_final,
                                    output["reason_logits_primary"][p,label], q_primary, weight))
            gains.append((output["reason_logits_final_train"][p,label]-q_final)-
                         (output["reason_logits_primary"][p,label]-q_primary).detach()); pairs += count; pair_labels[label] += int(count)
        if negatives.numel() and queued_pos:
            count = min(negatives.numel(), len(queued_pos)); n = negatives[:count]
            q_final = torch.stack([row["final_reason_logits"][label] for row in queued_pos[:count]]).to(output["reason_delta"])
            q_primary = torch.stack([row["primary_reason_logits"][label] for row in queued_pos[:count]]).to(output["reason_delta"])
            age = output["reason_delta"].new_tensor([update-row["enqueue_update"] for row in queued_pos[:count]])
            weight = structured["reason_counter_reliability"][n,label] * torch.exp(-age / queue.age_tau)
            losses.append(ecpo_loss(q_final, output["reason_logits_final_train"][n,label], q_primary,
                                    output["reason_logits_primary"][n,label], weight))
            gains.append((q_final-output["reason_logits_final_train"][n,label])-
                         (q_primary-output["reason_logits_primary"][n,label]).detach()); pairs += count; pair_labels[label] += int(count)
    zero = output["reason_delta"].sum() * 0.0
    return (torch.stack(losses).mean() if losses else zero,
            torch.cat(gains) if gains else None, pairs, pair_labels)


def compute_loss(output, batch, structured, cfg, schedule, cf, dual, ecpo_pack):
    w = cfg["loss_weights"]; reg = AIECertLossRegistry(); action, reason = batch["action"], batch["reason"]
    neg_weight = 0.25 + 0.75 * structured["reason_counter_reliability"].detach()
    reg.add("primary_action", "primary_core", asymmetric_loss_with_logits(output["action_logits_primary"], action), w["primary_action"])
    reg.add("primary_action_visual", "primary_core", asymmetric_loss_with_logits(output["action_visual_logits_primary"], action), w["primary_action_visual"])
    reg.add("primary_action_reason", "primary_core", asymmetric_loss_with_logits(output["action_reason_logits_primary"], action), w["primary_action_reason"])
    reg.add("primary_reason_partial", "primary_core", partial_asl(output["reason_logits_primary"], reason, neg_weight), w["primary_reason_partial"])
    reg.add("primary_reason_soft_f1", "primary_core", soft_f1(output["reason_logits_primary"], reason, neg_weight), w["primary_reason_soft_f1"])
    pred_target, pred_pos, pred_counter = structured["predicate_target"], structured["predicate_positive_mask"], structured["predicate_counter_mask"]
    pred_mask = (pred_pos + pred_counter).clamp_max(1.0)
    pred_loss = F.binary_cross_entropy_with_logits(output["predicate_logits_clean"], pred_target, reduction="none")
    pred_loss = (pred_loss * pred_mask * structured["predicate_reliability"].clamp_min(0.25)).sum() / pred_mask.sum().clamp_min(1.0)
    reg.add("predicate_cls", "predicate_visual", schedule["grounding_scale"] * pred_loss, w["predicate_cls"])
    map_error = (output["predicate_attention_clean"] - structured["predicate_map_target"]).abs().sum(-1)
    map_loss = (map_error * structured["predicate_map_mask"]).sum() / structured["predicate_map_mask"].sum().clamp_min(1.0)
    reg.add("predicate_map", "predicate_visual", schedule["grounding_scale"] * map_loss, w["predicate_map"])
    compact = output["predicate_attention_clean"].sqrt().sum(-1).mean()
    reg.add("predicate_compactness", "predicate_visual", compact, w["predicate_compactness"])
    reg.add("final_action", "action_contribution", partial_asl(output["action_logits_final_train"], action), w["final_action"])
    reg.add("final_action_soft_f1", "action_contribution", soft_f1(output["action_logits_final_train"], action), w["final_action_soft_f1"])
    reg.add("final_action_cardinality", "action_contribution", F.smooth_l1_loss(torch.sigmoid(output["action_logits_final_train"]).sum(-1), action.sum(-1)), w["final_action_cardinality"])
    reg.add("atom_overlap_ceiling", "action_evidence", atom_overlap_ceiling_loss(output["atom_map"], output["bounded_contribution"]), w["atom_overlap_ceiling"])
    reg.add("final_reason", "reason_private", partial_asl(output["reason_logits_final_train"], reason, neg_weight), w["final_reason"])
    reg.add("final_reason_soft_f1", "reason_private", soft_f1(output["reason_logits_final_train"], reason, neg_weight), w["final_reason_soft_f1"])
    ecpo_value, ecpo_gain, pair_count, _ = ecpo_pack
    reg.add("ecpo", "reason_private", schedule["ecpo_scale"] * ecpo_value, w["ecpo"], pair_count > 0, "no_verified_pair" if not pair_count else "")
    positive_names = structured["predicate_positive_mask"][:, None, None].bool().expand_as(output["name_quality"])
    name_loss = naming_preference_loss(output["name_quality"], positive_names)
    reg.add("naming_preference", "naming_readout", schedule["naming_scale"] * name_loss, w["naming_preference"], bool(positive_names.any()), "no_grounded_name")
    constraints, availability = evidence_constraints(output, cf, ecpo_gain)
    reg.add("primal_dual", "action_evidence", schedule["dual_scale"] * dual.primal_loss(constraints), 1.0)
    return reg.total(), reg, constraints, availability, pair_count


@torch.no_grad()
def collect_predictions(model, loader, device, schedule, audit_limit=0):
    model.eval(); store = {k: [] for k in ("ap", "af", "rp", "rf", "at", "rt", "action_delta", "reason_delta", "reason_budget")}; names=[]
    variant_defs = {"final": {}, "predicate_prior_off": {"predicate_prior_scale": 0.0},
        "local_reread_off": {"local_reread_enabled": False}, "atom_transport_off": {"transport_enabled": False},
        "background_center_off": {"background_center_enabled": False}, "action_residual_off": {"action_residual_enabled": False},
        "reason_action_prior_off": {"reason_action_prior_enabled": False}, "reason_predicate_prior_off": {"reason_predicate_prior_enabled": False},
        "reason_signed_to_unsigned_legacy": {"reason_signed_priors": False}, "reason_budget_off": {"reason_budget_enabled": False},
        "reason_delta_off": {"reason_delta_enabled": False}}
    audit_store={name:{"action":[],"reason":[]} for name in variant_defs}; audit_at=[]; audit_rt=[]; audited=0
    for batch in loader:
        field=model.encode_images(batch["image"].to(device, non_blocking=True))
        base=dict(action_scale=schedule["action_scale"],reason_budget_max=schedule["reason_budget_max"],
                  predicate_prior_scale=schedule["predicate_prior_scale"],transport_gamma_cap=schedule["transport_gamma_cap"])
        out = model.decode_from_field(field,**base)
        for key, value in (("ap", out["action_logits_primary"]), ("af", out["action_logits_final"]),
                           ("rp", out["reason_logits_primary"]), ("rf", out["reason_logits_final"])):
            store[key].append(value.float().cpu())
        store["at"].append(batch["action"]); store["rt"].append(batch["reason"])
        store["action_delta"].append(out["action_delta"].float().cpu())
        store["reason_delta"].append(out["reason_delta"].float().cpu())
        store["reason_budget"].append(out["reason_budget"].float().cpu()); names.extend(batch["file_name"])
        if audited<audit_limit:
            take=min(batch["image"].shape[0],audit_limit-audited)
            sliced={key:(value[:take] if torch.is_tensor(value) and value.ndim and value.shape[0]==batch["image"].shape[0] else value) for key,value in field.items()}
            for name,overrides in variant_defs.items():
                branch=model.decode_from_field(sliced,**{**base,**overrides})
                audit_store[name]["action"].append(branch["action_logits_final"].float().cpu()); audit_store[name]["reason"].append(branch["reason_logits_final"].float().cpu())
            audit_at.append(batch["action"][:take]); audit_rt.append(batch["reason"][:take]); audited+=take
    result={k: torch.cat(v) for k,v in store.items()}; result["file_names"]=names
    if audited:
        at,rt=torch.cat(audit_at),torch.cat(audit_rt); audit_metrics={}
        for name,value in audit_store.items():
            value["action"],value["reason"]=torch.cat(value["action"]),torch.cat(value["reason"])
            audit_metrics[name]=branch_metrics(value["action"],value["reason"],at,rt)
        result["fixed_audit"]={"metrics":audit_metrics,"logits":audit_store,"action_target":at,"reason_target":rt}
    return result


@torch.no_grad()
def evaluate(model, calib_loader, test_loader, device, schedule, calibration, cfg):
    calib = collect_predictions(model, calib_loader, device, schedule)
    raw_calib = branch_metrics(calib["af"], calib["rf"], calib["at"], calib["rt"])
    candidate = fit_posthoc_thresholds(torch.cat((calib["af"], calib["rf"]), 1),
        torch.cat((calib["at"], calib["rt"]), 1), [list(range(4)), list(range(4,25))],
        shrinkage_support=cfg["calibration"]["group_shrinkage_support"], grid_step=cfg["calibration"]["grid_step"])
    candidate_prob = candidate["threshold_prob"].float()
    candidate_logits = apply_posthoc_threshold(torch.cat((calib["af"], calib["rf"]), 1), candidate_prob)
    candidate_metrics = branch_metrics(candidate_logits[:,:4], candidate_logits[:,4:], calib["at"], calib["rt"])
    guard = calibration.propose(candidate_prob, raw_calib["joint"], raw_calib["Act_mF1"],
                                candidate_metrics["joint"], candidate_metrics["Act_mF1"])
    accepted = guard["accepted_threshold"].float()
    test = collect_predictions(model, test_loader, device, schedule, int(cfg["runtime"]["fixed_test_audit_samples"]))
    deploy_logits = apply_posthoc_threshold(torch.cat((test["af"], test["rf"]), 1), accepted)
    primary_deploy_logits = apply_posthoc_threshold(torch.cat((test["ap"], test["rp"]), 1), accepted)
    metrics = {"primary": branch_metrics(test["ap"], test["rp"], test["at"], test["rt"]),
               "final_raw": branch_metrics(test["af"], test["rf"], test["at"], test["rt"]),
               "primary_deploy": branch_metrics(primary_deploy_logits[:,:4], primary_deploy_logits[:,4:], test["at"], test["rt"]),
               "deploy": branch_metrics(deploy_logits[:,:4], deploy_logits[:,4:], test["at"], test["rt"]),
               "calibration_guard": {**guard, "raw_calib": raw_calib, "candidate_calib": candidate_metrics}}
    test["ad"], test["rd"] = deploy_logits[:,:4], deploy_logits[:,4:]
    return metrics, test


def save_checkpoint(path, model, optimizer, dual, queue, calibration, epoch, update, total_updates, best, cfg):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "dual": dual.state_dict(),
        "preference_queue": queue.state_dict(), "calibration_guard": calibration.state_dict(), "epoch": epoch,
        "optimizer_update": update, "schedule_total_updates": total_updates, "best": best, "config": cfg,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-kind", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--epochs", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--num-workers", type=int); parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-calib-samples", type=int)
    parser.add_argument("--max-audit-samples", type=int); parser.add_argument("--max-test-samples", type=int); parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume"); parser.add_argument("--use-mock-dino", action="store_true")
    args = parser.parse_args(); cfg = load_config(args.config); outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["data"]["split_seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device); model = build_model(cfg, device, args.use_mock_dino)
    optimizer = build_optimizer(model, cfg); dual = AIECertDualState(cfg["dual"]["lr"], cfg["dual"]["ema_decay"], cfg["dual"]["lambda_max"]).to(device)
    queue = AIECertPreferenceQueue(cfg["ecpo"]["queue_capacity"], cfg["ecpo"]["max_age_updates"], cfg["ecpo"]["age_tau"])
    calibration = AIECertCalibrationGuard(25, cfg["calibration"]["ema_decay"], cfg["calibration"]["max_threshold_step"])
    epochs = args.epochs or cfg["training"]["epochs"]; batch_size = args.batch_size or cfg["training"]["batch_size"]
    accumulation = args.gradient_accumulation_steps or cfg["training"]["gradient_accumulation_steps"]
    workers = cfg["data"]["num_workers"] if args.num_workers is None else args.num_workers
    train_all, test = make_dataset(cfg, "train"), make_dataset(cfg, "test")
    calib_count = max(1, int(len(train_all) * cfg["data"]["train_calib_fraction"]))
    calib_indices = list(range(len(train_all) - calib_count, len(train_all)))
    if args.max_calib_samples: calib_indices = calib_indices[:args.max_calib_samples]
    calib = Subset(train_all, calib_indices)
    audit = Subset(train_all, range(min(args.max_audit_samples, len(train_all)))) if args.max_audit_samples else None
    train = train_all
    if args.max_train_samples: train = Subset(train, range(min(args.max_train_samples, len(train))))
    if args.max_test_samples: test = Subset(test, range(min(args.max_test_samples, len(test))))
    train_loader = make_loader(train, batch_size, True, workers, cfg)
    calib_loader = make_loader(calib, batch_size, False, workers, cfg)
    audit_loader = make_loader(audit, batch_size, False, workers, cfg) if audit is not None else None
    test_loader = make_loader(test, batch_size, False, workers, cfg)
    builder = AIECertStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"], cfg["primary"]["reason_counter_evidence"], cfg["data"]["bdd100k_root"])
    total_updates = math.ceil(len(train_loader) / accumulation) * epochs
    epoch_start = update = 0
    best = {"deploy_joint": -1.0, "action_mf1": -1.0, "action_map": -1.0,
            "reason_mf1": -1.0, "reason_map": -1.0}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device); model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        dual.load_state_dict(ckpt["dual"]); queue.load_state_dict(ckpt["preference_queue"]); calibration.load_state_dict(ckpt["calibration_guard"])
        epoch_start, update, total_updates, best = ckpt["epoch"] + 1, ckpt["optimizer_update"], ckpt["schedule_total_updates"], ckpt["best"]
    write_json(outdir / "run_manifest.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "run_kind": args.run_kind, "config": cfg, "test_only_epoch_eval": True, "feature_cache_enabled": False,
        "token_compression": "none", "schedule_total_updates": total_updates})
    (outdir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(outdir / "implementation_fingerprint.json", {"git_head": subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "formal_model": "AIECertOIAModel", "source_head": cfg["experiment"]["source_head"]})
    write_json(outdir / "split_manifest.json", {"seed": seed, "train_count": len(train),
        "train_calib_count": len(calib), "train_audit_count": len(audit) if audit is not None else 0,
        "test_count": len(test)})
    write_json(outdir / "best_checkpoints.json", best)
    for epoch in range(epoch_start, epochs):
        model.train(); dual.train(); optimizer.zero_grad(set_to_none=True); last_diag = {}; last_owner = {}
        cf_count = ecpo_count = cf_positive = cf_total = 0
        cf_control_types = torch.zeros(4, dtype=torch.long)
        cf_contributions, cf_certificates = [], []
        ecpo_label_counts = torch.zeros(21, dtype=torch.long)
        for micro, batch in enumerate(train_loader):
            schedule = schedule_values(update, total_updates, cfg)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * schedule["lr_multiplier"]
            images, action, reason = batch["image"].to(device, non_blocking=True), batch["action"].to(device), batch["reason"].to(device)
            structured = builder.build(batch["file_name"], device=device)
            flush = (micro + 1) % accumulation == 0 or micro + 1 == len(train_loader)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                field = model.encode_images(images)
                output = model.decode_from_field(field, action_scale=schedule["action_scale"], reason_budget_max=schedule["reason_budget_max"],
                    predicate_prior_scale=schedule["predicate_prior_scale"], transport_gamma_cap=schedule["transport_gamma_cap"])
                cf = run_counterfactual(model, field, output, action, schedule) if (flush and schedule["cf_scale"] > 0 and
                    update % cfg["counterfactual"]["interval_optimizer_updates"] == 0) else None
                ecpo_pack = build_ecpo(output, {"reason": reason}, structured, update, queue, cfg["ecpo"]["verified_counter_threshold"], cfg["ecpo"]["pairs_per_label"])
                loss, registry, constraints, availability, pairs = compute_loss(output, {"action": action, "reason": reason}, structured, cfg, schedule, cf, dual, ecpo_pack)
            divisor = accumulation_divisor(micro, len(train_loader), accumulation)
            (loss / divisor).backward(); ecpo_count += pairs; ecpo_label_counts += ecpo_pack[3]
            queue.enqueue(PreferenceBatch(output["reason_logits_primary"], output["reason_logits_final"], reason,
                structured["reason_verified_counter"], structured["reason_counter_reliability"],
                torch.full((images.shape[0],), update, device=device), batch["file_name"]))
            if flush:
                print_due = (update + 1) % cfg["runtime"]["print_every_optimizer_updates"] == 0
                owner_before_clip = owner_gradient_stats(model)
                parameter_snapshot = snapshot_owner_parameters(model) if print_due else None
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["global_grad_clip"])
                owner_after_clip = owner_gradient_stats(model)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); dual.update({k: v for k, v in constraints.items() if availability.get(k)}, schedule["dual_scale"]); update += 1
                last_diag = evidence_diagnostics(output)
                last_owner = {"before_clip": owner_before_clip, "after_clip": owner_after_clip,
                              "update_rms": owner_update_rms(model, parameter_snapshot) if parameter_snapshot else {}}
                if cf is not None:
                    valid = cf["valid_mask"].detach().cpu()
                    certificates = cf["certificate"].detach().cpu()[valid]
                    cf_count += int(valid.sum()); cf_total += int(certificates.numel())
                    cf_positive += int((certificates > 0).sum())
                    cf_control_types += cf["per_control_validity"].detach().cpu().any(0).long()
                    rows = torch.arange(output["bounded_contribution"].shape[0], device=device)
                    chosen = output["bounded_contribution"][rows,cf["action_id"],cf["atom_id"]].detach().cpu()[valid]
                    cf_contributions.append(chosen); cf_certificates.append(certificates)
                if update % cfg["runtime"]["print_every_optimizer_updates"] == 0:
                    event={"aie_cert_batch":True,"epoch":epoch,"micro_step":micro,"optimizer_update":update,
                        "progress":update/max(total_updates,1),"lr_by_owner":{g["name"]:g["lr"] for g in optimizer.param_groups},
                        **schedule,"loss_total":float(loss.detach()),"losses":{n:{"raw":float(t.value.detach()),"weighted":float((t.value*t.weight).detach()),"owner":t.owner,"active":t.active,"reason":t.inactivity_reason} for n,t in registry.terms.items()},
                        "constraints":{k:float(v.detach()) for k,v in constraints.items()},**last_diag,
                        "cf_valid":cf_count,"ecpo_pairs":ecpo_count,"dual":{n:float(getattr(dual,f"lambda_{n}")) for n in dual.NAMES},
                        "owner_gradients":last_owner,"cf_certificate_positive_rate":cf_positive/max(cf_total,1),
                        "cf_control_types_observed":int((cf_control_types>0).sum()),
                        "gpu_allocated_gb":torch.cuda.memory_allocated(device)/1024**3 if device.type=="cuda" else 0,
                        "gpu_reserved_gb":torch.cuda.memory_reserved(device)/1024**3 if device.type=="cuda" else 0}
                    print(event,flush=True); append_jsonl(outdir/"mechanism_stats.jsonl",event)
                append_jsonl(outdir/"loss_components.jsonl",{"epoch":epoch,"optimizer_update":update,"total":float(loss.detach()),
                    "terms":{n:float(t.value.detach()) for n,t in registry.terms.items()}})
        schedule = schedule_values(update, total_updates, cfg)
        epoch_dir = outdir / f"epoch_{epoch:03d}"; epoch_dir.mkdir(exist_ok=True)
        save_checkpoint(epoch_dir/"checkpoint_pre_eval.pth",model,optimizer,dual,queue,calibration,epoch,update,total_updates,best,cfg)
        metrics, tensors = evaluate(model, calib_loader, test_loader, device, schedule, calibration, cfg)
        train_audit_metrics = None
        if audit_loader is not None:
            audit_tensors = collect_predictions(model, audit_loader, device, schedule)
            train_audit_metrics = {"primary":branch_metrics(audit_tensors["ap"],audit_tensors["rp"],audit_tensors["at"],audit_tensors["rt"]),
                                   "final":branch_metrics(audit_tensors["af"],audit_tensors["rf"],audit_tensors["at"],audit_tensors["rt"])}
            write_json(epoch_dir/"train_audit_metrics.json",train_audit_metrics)
        write_json(epoch_dir / "metrics.json", metrics); write_json(epoch_dir / "branch_metrics.json", metrics)
        if sum(value.numel() for value in cf_certificates) >= 2:
            cert_values, contribution_values = torch.cat(cf_certificates), torch.cat(cf_contributions)
            correlation = torch.corrcoef(torch.stack((contribution_values,cert_values)))[0,1]
            cf_correlation = float(correlation) if torch.isfinite(correlation) else None
        else:
            cf_correlation = None
        cf_summary = {"valid_events":cf_count,"certificate_positive_rate":cf_positive/max(cf_total,1),
                      "control_types_observed":int((cf_control_types>0).sum()),
                      "contribution_certificate_pearson":cf_correlation}
        write_json(epoch_dir / "schedule.json", schedule); write_json(epoch_dir / "owner_gradients.json", {"exact_cover": True, **last_owner})
        write_json(epoch_dir / "evidence_diagnostics.json", last_diag); write_json(epoch_dir / "counterfactual_diagnostics.json", cf_summary)
        queue_ages = [update - int(record["enqueue_update"]) for record in queue.records]
        ecpo_summary = {"valid_pairs":ecpo_count,"queue_size":len(queue.records),
                        "queue_max_age":max(queue_ages,default=0),
                        "labels_with_pairs":int((ecpo_label_counts>0).sum()),"per_label_pairs":ecpo_label_counts.tolist()}
        write_json(epoch_dir / "dual_state.json", dual.state_dict()); write_json(epoch_dir / "ecpo_diagnostics.json", ecpo_summary)
        write_json(epoch_dir / "naming_diagnostics.json", {"named_coverage": float(output["named_coverage"].detach())})
        write_json(epoch_dir / "calibration_guard.json", {"source": "train_calib", "test_writeback": False, **metrics["calibration_guard"]})
        write_json(epoch_dir/"metrics_primary_raw.json",metrics["primary"]); write_json(epoch_dir/"metrics_final_raw.json",metrics["final_raw"])
        write_json(epoch_dir/"metrics_primary_deploy.json",metrics["primary_deploy"]); write_json(epoch_dir/"metrics_final_deploy.json",metrics["deploy"])
        write_json(epoch_dir/"per_action_metrics.json",{k:v for k,v in metrics["deploy"].items() if "per_label" in k and k.startswith("Act")})
        write_json(epoch_dir/"per_reason_metrics.json",{k:v for k,v in metrics["deploy"].items() if "per_label" in k and k.startswith("Exp")})
        write_json(epoch_dir/"calibration.json",metrics["calibration_guard"]); write_json(epoch_dir/"mechanism_summary.json",last_diag)
        write_json(epoch_dir/"predicate_mixture_stats.json",{k:v for k,v in last_diag.items() if "predicate" in k})
        write_json(epoch_dir/"atom_transport_stats.json",{k:v for k,v in last_diag.items() if "transport" in k})
        write_json(epoch_dir/"contribution_stats.json",{"reconstruction_error":last_diag.get("contribution_reconstruction_error")})
        write_json(epoch_dir/"counterfactual_certificate.json",cf_summary)
        write_json(epoch_dir/"dual_constraints.json",{"state":dual.state_dict(),"availability":availability})
        write_json(epoch_dir/"reason_budget_stats.json",{k:v for k,v in last_diag.items() if "reason" in k})
        write_json(epoch_dir/"ecpo_stats.json",ecpo_summary)
        write_json(epoch_dir/"naming_stats.json",{"named_coverage":float(output["named_coverage"].detach())})
        write_json(epoch_dir/"structured_coverage.json",structured["coverage"])
        audit_metrics = tensors["fixed_audit"]["metrics"]
        write_json(epoch_dir/"branch_audit_128.json",audit_metrics)
        write_json(epoch_dir/"component_diagnosis.json",component_diagnosis(audit_metrics,last_diag))
        test_output={"file_names":tensors["file_names"],"action_target":tensors["at"],"reason_target":tensors["rt"],
            "action_logits_primary":tensors["ap"],"action_logits_final":tensors["af"],"reason_logits_primary":tensors["rp"],
            "reason_logits_final":tensors["rf"],"accepted_threshold":metrics["calibration_guard"]["accepted_threshold"],
            "action_delta":tensors["action_delta"],"reason_delta":tensors["reason_delta"],"reason_budget":tensors["reason_budget"]}
        torch.save(test_output,epoch_dir/"test_outputs.pt"); torch.save(tensors["fixed_audit"],epoch_dir/"fixed_audit_outputs.pt")
        append_jsonl(outdir/"counterfactual_summary.jsonl",{"epoch":epoch,**cf_summary})
        append_jsonl(outdir/"dual_state.jsonl",{"epoch":epoch,"state":dual.state_dict()})
        append_jsonl(outdir/"ecpo_summary.jsonl",{"epoch":epoch,**ecpo_summary})
        append_jsonl(outdir/"runtime_stats.jsonl",{"epoch":epoch,"gpu_peak_gb":torch.cuda.max_memory_reserved(device)/1024**3 if device.type=="cuda" else 0})
        missing = validate_epoch(epoch_dir)
        if missing: raise RuntimeError(f"missing epoch artifacts: {missing}")
        append_jsonl(outdir / "metrics_summary.jsonl", {"epoch": epoch, **metrics["deploy"], "final_raw": metrics["final_raw"], "primary": metrics["primary"]})
        candidates = {"deploy_joint": (metrics["deploy"]["joint"], "checkpoint_best_deploy_joint.pth"),
                      "action_mf1": (metrics["deploy"]["Act_mF1"], "checkpoint_best_action_mf1.pth"),
                      "action_map": (metrics["deploy"]["Act_mAP"], "checkpoint_best_action_map.pth"),
                      "reason_mf1": (metrics["deploy"]["Exp_mF1"], "checkpoint_best_reason_mf1.pth"),
                      "reason_map": (metrics["deploy"]["Exp_mAP"], "checkpoint_best_reason_map.pth")}
        for key, (score, filename) in candidates.items():
            if score > best.get(key, -1.0):
                best[key] = score; best[f"{key}_epoch"] = epoch
                save_checkpoint(outdir / filename, model, optimizer, dual, queue, calibration,
                                epoch, update, total_updates, best, cfg)
        write_json(outdir/"best_checkpoints.json",best)
        save_checkpoint(outdir / f"checkpoint_epoch_{epoch:03d}.pth", model, optimizer, dual, queue,
                        calibration, epoch, update, total_updates, best, cfg)
        save_checkpoint(outdir / "checkpoint_latest.pth", model, optimizer, dual, queue, calibration, epoch, update, total_updates, best, cfg)
        print({"aie_cert_epoch": epoch, "Act_mF1": metrics["deploy"]["Act_mF1"], "Act_oF1": metrics["deploy"]["Act_oF1"],
               "Exp_mF1": metrics["deploy"]["Exp_mF1"], "Exp_oF1": metrics["deploy"]["Exp_oF1"], "best": best}, flush=True)


if __name__ == "__main__": main()
