from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.aie_splits import stable_split_ids, write_split_manifest
from fate_oia.datasets.aie_structured_evidence import AIEStructuredEvidenceBuilder
from fate_oia.engine.train_aie_oia import canonical_model_state_dict, collate, make_dataset
from fate_oia.losses import acpr_losses
from fate_oia.losses.aie_losses import (
    action_cardinality_loss, counterfactual_necessity_loss, predicate_map_compactness_loss, predicate_map_loss,
    predicate_masked_asl_loss, predicate_reason_alignment_pu_loss, probe_duplicate_loss, soft_f1_loss,
)
from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits
from fate_oia.losses.pact_loss_registry import PACTLossRegistry, exact_pact_owner_groups
from fate_oia.losses.pact_rank_losses import action_rank_trust_region, labelwise_reason_rank
from fate_oia.models.pact_oia_model import PACTOIAModel
from fate_oia.utils.aie_calibration import apply_posthoc_threshold, fit_posthoc_thresholds
from fate_oia.utils.aie_metrics import aie_branch_metrics, counterfactual_case_metrics
from fate_oia.utils.aie_counterfactual import AIECounterfactualConfig, AIECounterfactualEngine
from fate_oia.utils.pact_artifacts import append_jsonl, sha256, write_json
from fate_oia.utils.pact_pair_queue import PACTBalancedPairQueue
from fate_oia.utils.pact_pareto_controller import PACTParetoController


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, cfg: dict, generator=None) -> DataLoader:
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                  pin_memory=bool(cfg["data"]["pin_memory"]), generator=generator,
                  persistent_workers=bool(cfg["data"]["persistent_workers"]) and workers > 0)
    if workers:
        kwargs["prefetch_factor"] = int(cfg["data"]["prefetch_factor"])
    return DataLoader(dataset, **kwargs)


def build_model(cfg: dict, device: torch.device) -> PACTOIAModel:
    p, b, e = cfg["primary"], cfg["backbone"], cfg["evidence"]
    return PACTOIAModel(
        dim=int(p["dim"]), selected_layers=tuple(b["selected_layers"]), pretrained_weights=b["pretrained_weights"],
        scene_config=p["scene_predicates"], grammar_path=p["reason_grammar"],
        probes_per_action=int(e["probes_per_action"]), local_points_per_layer=int(e["local_points_per_layer"]),
        max_offset=float(e["max_offset"]), predicate_bias_max=float(e["predicate_bias_max"]),
        probe_chunk_size=int(e["probe_chunk_size"]), action_kappa=float(e["action_kappa"]),
        reason_kappa=float(cfg["reason"]["kappa"]),
    ).to(device)


def load_source(model: PACTOIAModel, checkpoint_path: str | Path) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    migration = model.migrate_from_aie_state_dict(state)
    return {"checkpoint_epoch": int(checkpoint.get("epoch", -1)), "migration": migration}


def build_optimizer(model: PACTOIAModel, cfg: dict) -> torch.optim.Optimizer:
    owners = exact_pact_owner_groups(model)
    groups = [{"params": parameters, "name": name, "base_lr": float(cfg["training"][f"lr_{name}"]),
               "lr": float(cfg["training"][f"lr_{name}"])} for name, parameters in owners.items()]
    return torch.optim.AdamW(groups, weight_decay=float(cfg["training"]["weight_decay"]))


def cosine_schedule(update: int, total: int, cfg: dict) -> float:
    progress = update / max(total, 1)
    warmup = float(cfg["training"]["warmup_ratio"])
    if progress < warmup:
        return max(progress / max(warmup, 1e-8), 1e-3)
    ratio = (progress - warmup) / max(1.0 - warmup, 1e-8)
    minimum = float(cfg["training"]["min_lr_ratio"])
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def counterfactual_due(mode: str, update: int, micro: int, accumulation: int, interval: int) -> bool:
    return mode == "pact" and micro + 1 == accumulation and (update + 1) % interval == 0


def _action_pairs(final: Tensor, primary: Tensor, target: Tensor) -> dict[str, Tensor]:
    final_pos, final_neg, primary_pos, primary_neg = [], [], [], []
    for label in range(target.shape[1]):
        pos, neg = target[:, label] > 0.5, target[:, label] <= 0.5
        if pos.any() and neg.any():
            count = min(int(pos.sum()), int(neg.sum()))
            final_pos.append(final[pos, label][:count]); final_neg.append(final[neg, label][:count])
            primary_pos.append(primary[pos, label][:count]); primary_neg.append(primary[neg, label][:count])
    if not final_pos:
        zero = final.sum() * 0
        return action_rank_trust_region(zero[None], zero[None], zero.detach()[None], zero.detach()[None])
    return action_rank_trust_region(torch.cat(final_pos), torch.cat(final_neg),
                                    torch.cat(primary_pos), torch.cat(primary_neg))


def _reason_rank_loss(logits: Tensor, targets: Tensor, queue: PACTBalancedPairQueue, update: int) -> tuple[Tensor, dict]:
    references, stats = queue.pairs(update, logits.device)
    losses = []
    for label, historical_pos, historical_neg in references:
        pos, neg = targets[:, label] > 0.5, targets[:, label] <= 0.5
        if pos.any():
            count = min(int(pos.sum()), historical_neg.numel())
            losses.append(labelwise_reason_rank(logits[pos, label][:count], historical_neg[:count]))
        if neg.any():
            count = min(int(neg.sum()), historical_pos.numel())
            losses.append(labelwise_reason_rank(historical_pos[:count], logits[neg, label][:count]))
    return (torch.stack(losses).mean() if losses else logits.sum() * 0), stats


def compute_losses(output: dict, batch: dict, structured: dict, cfg: dict,
                   queue: PACTBalancedPairQueue, update: int, cf: dict | None = None,
                   cf_accumulation_compensation: float = 1.0) -> tuple[Tensor, list[dict], dict]:
    registry = PACTLossRegistry(cfg["loss_weights"])
    action, reason = batch["action"], batch["reason"]
    negative_weight = torch.where(reason > 0.5, torch.ones_like(reason), 0.25 + 0.75 * output["contradiction_score"].detach())
    registry.add("primary_action", "context_action", asymmetric_loss_with_logits(output["action_logits_primary"], action))
    registry.add("primary_action_visual", "context_action", asymmetric_loss_with_logits(output["action_visual_logits_primary"], action))
    registry.add("primary_action_context", "context_action", asymmetric_loss_with_logits(output["action_reason_logits_primary"], action))
    registry.add("formal_reason_partial", "explanation_lane", acpr_losses.partial_label_reason_loss(
        output["reason_logits_primary"], reason, output["contradiction_score"]))
    formal_rank, queue_stats = _reason_rank_loss(output["reason_logits_primary"], reason, queue, update)
    registry.add("formal_reason_rank", "explanation_lane", formal_rank)
    registry.add("formal_reason_soft_f1", "explanation_lane", soft_f1_loss(output["reason_logits_primary"], reason, negative_weight))
    registry.add("predicate_cls", "predicate_visual", predicate_masked_asl_loss(
        output["semantic_predicate_logits"], structured["predicate_target"], structured["predicate_target_mask"],
        structured["predicate_counter_mask"], structured["predicate_reliability"]))
    registry.add("predicate_map", "predicate_visual", predicate_map_loss(
        output["semantic_predicate_attention"], structured["predicate_map_target"], structured["predicate_map_mask"]))
    registry.add("predicate_reason_align", "explanation_lane", predicate_reason_alignment_pu_loss(
        output["semantic_predicate_probs"], reason, output["_grammar_positive_mask"],
        output["_grammar_contradictory_mask"], negative_weight))
    registry.add("predicate_compactness", "predicate_visual", predicate_map_compactness_loss(output["semantic_predicate_attention"]))
    registry.add("final_action", "action_contribution", asymmetric_loss_with_logits(output["action_logits_final_train"], action))
    registry.add("final_action_soft_f1", "action_contribution", soft_f1_loss(output["action_logits_final_train"], action))
    registry.add("final_action_cardinality", "action_contribution", action_cardinality_loss(output["action_logits_final_train"], action))
    registry.add("final_reason", "reason_private", acpr_losses.partial_label_reason_loss(
        output["reason_logits_final_train"], reason, output["contradiction_score"]))
    rank = _action_pairs(output["action_logits_final_train"], output["action_logits_primary"].detach(), action)
    registry.add("action_rank_repair", "action_contribution", rank["repair_loss"])
    registry.add("action_rank_preserve", "action_contribution", rank["preserve_loss"])
    reason_rank, _ = _reason_rank_loss(output["reason_logits_final_train"], reason, queue, update)
    registry.add("final_reason_rank", "reason_private", reason_rank)
    registry.add("final_reason_soft_f1", "reason_private", soft_f1_loss(output["reason_logits_final_train"], reason, negative_weight))
    zero = output["action_logits_final_train"].sum() * 0
    if cf and int(cf.get("cf_valid_count", 0)) > 0:
        valid = cf["valid_mask"]
        registry.add("cf_necessity", "action_evidence", counterfactual_necessity_loss(
            cf["selected_drop"], cf["control_drop"], valid) * float(cf_accumulation_compensation))
        registry.add("cf_sufficiency", "action_evidence", (cf["sufficiency_loss_raw"] * valid).sum() /
                     valid.sum().clamp_min(1) * float(cf_accumulation_compensation))
    else:
        registry.add("cf_necessity", "action_evidence", zero)
        registry.add("cf_sufficiency", "action_evidence", zero)
    registry.add("probe_duplicate", "action_evidence", probe_duplicate_loss(output["evidence_map"], output["bounded_contribution"], action))
    registry.add("action_delta", "action_contribution", output["action_delta"].square().mean())
    return registry.total(), registry.rows(), {**rank, **queue_stats}


def pareto_controller_step(model: PACTOIAModel, controller: PACTParetoController, train_output: dict,
                           train_batch: dict, audit_batch: dict, cfg: dict, device: torch.device) -> dict:
    shared_parameters = dict(model.shared_readout.named_parameters())
    parameters = tuple(shared_parameters.values())
    # The ordinary forward produces BF16 nodes. Re-enter the same autocast
    # policy here because this controller runs outside the training context.
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        action_objective = asymmetric_loss_with_logits(train_output["action_logits_primary"], train_batch["action"])
        # Build the semantic objective directly from shared nodes so g_S is measured at license=1.
        semantic = model.explanation_decoder(
            train_output["shared_label_nodes"], train_output["predicate_tokens"].detach()
        )
        semantic_reason = model.predicate_reason(
            semantic["reason_nodes_formal"], train_output["semantic_predicate_probs"].detach(),
            train_output["predicate_tokens"].detach(),
        )
        semantic_logits = semantic["reason_logits_visual_formal"] + semantic_reason["predicate_reason_delta"]
        semantic_objective = acpr_losses.partial_label_reason_loss(
            semantic_logits, train_batch["reason"], semantic_reason["contradiction_score"]
        )
    action_grad = torch.autograd.grad(action_objective, parameters, retain_graph=True, allow_unused=True)
    semantic_grad = torch.autograd.grad(semantic_objective, parameters, retain_graph=True, allow_unused=True)
    action_grad = tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, action_grad))
    semantic_grad = tuple(torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, semantic_grad))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        audit_images = audit_batch["image"].to(device, non_blocking=True)
        audit_target = audit_batch["action"].to(device, non_blocking=True)
        field = model.encode_images(audit_images)
        raw = field["patch_tokens_by_layer"]
        patch0, _, masks, _ = model.ego(raw[:, 0])
        audit_patch = raw.clone(); audit_patch[:, 0] = patch0
        audit_predicate = model.predicate_head(audit_patch, region_masks=masks)["predicate_tokens"]
    meta_step = float(cfg["pareto"]["meta_step"])

    def evaluator(candidate: float) -> float:
        candidate_parameters = {
            name: parameter - meta_step * (ga + float(candidate) * gs)
            for (name, parameter), ga, gs in zip(shared_parameters.items(), action_grad, semantic_grad)
        }
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            shared = functional_call(model.shared_readout, candidate_parameters, (audit_patch,))
            action = model.context_decoder(shared["shared_label_nodes"], audit_predicate)["action_logits_primary"]
            return float(asymmetric_loss_with_logits(action, audit_target))

    result = controller.evaluate_candidates(evaluator, model)
    flat_a = torch.cat([value.flatten() for value in action_grad])
    flat_s = torch.cat([value.flatten() for value in semantic_grad])
    result["action_semantic_grad_cosine"] = float(torch.dot(flat_a, flat_s) / (flat_a.norm() * flat_s.norm()).clamp_min(1e-12))
    result["action_grad_norm"] = float(flat_a.norm()); result["semantic_grad_norm"] = float(flat_s.norm())
    return result


@torch.no_grad()
def collect(model: PACTOIAModel, loader: DataLoader, device: torch.device, mode: str, license_value: float,
            cfg: dict) -> dict[str, Tensor | list[str]]:
    model.eval(); store: dict[str, list] = {key: [] for key in (
        "action_primary", "action_final", "reason_primary", "reason_final", "action_target", "reason_target")}
    names = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(images, semantic_share_license=license_value,
                           action_scale=0.0 if mode == "control" else float(cfg["evidence"]["action_scale"]),
                           reason_budget=0.0 if mode == "control" else float(cfg["reason"]["selected_budget"]),
                           compatibility_mode=mode == "control")
        for key, value in (("action_primary", output["action_logits_primary"]), ("action_final", output["action_logits_final"]),
                           ("reason_primary", output["reason_logits_primary"]), ("reason_final", output["reason_logits_final"]),
                           ("action_target", batch["action"]), ("reason_target", batch["reason"])):
            store[key].append(value.detach().cpu())
        names.extend(batch["file_name"])
    return {**{key: torch.cat(value) for key, value in store.items()}, "file_name": names}


def evaluate(model, calib_loader, test_loader, device, mode, license_value, cfg, epoch_dir: Path) -> dict:
    calib = collect(model, calib_loader, device, mode, license_value, cfg)
    test = collect(model, test_loader, device, mode, license_value, cfg)
    thresholds = fit_posthoc_thresholds(
        torch.cat((calib["action_final"], calib["reason_final"]), 1),
        torch.cat((calib["action_target"], calib["reason_target"]), 1), [list(range(4)), list(range(4, 25))]
    )["threshold_prob"]
    primary = aie_branch_metrics(test["action_primary"], test["reason_primary"], test["action_target"], test["reason_target"])
    final = aie_branch_metrics(test["action_final"], test["reason_final"], test["action_target"], test["reason_target"])
    deploy = aie_branch_metrics(
        apply_posthoc_threshold(test["action_final"], thresholds[:4]),
        apply_posthoc_threshold(test["reason_final"], thresholds[4:]), test["action_target"], test["reason_target"])
    payload = {"primary": primary, "final_raw": final, "deploy": deploy,
               "thresholds_train_calib": thresholds.tolist()}
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(test, epoch_dir / "test_outputs.pt")
    for file_name, key in (
        ("action_logits_primary_test.pt", "action_primary"), ("action_logits_final_test.pt", "action_final"),
        ("reason_logits_primary_test.pt", "reason_primary"), ("reason_logits_final_test.pt", "reason_final"),
        ("labels_action_test.pt", "action_target"), ("labels_reason_test.pt", "reason_target"),
    ):
        torch.save(test[key], epoch_dir / file_name)
    (epoch_dir / "file_names_test.json").write_text(json.dumps(test["file_name"], ensure_ascii=False), encoding="utf-8")
    write_json(epoch_dir / "branch_metrics.json", payload)
    write_json(epoch_dir / "per_label_metrics.json", {
        "action_f1": final["Act_per_label_f1"], "action_ap": final["Act_per_label_ap"],
        "reason_f1": final["Exp_per_label_f1"], "reason_ap": final["Exp_per_label_ap"],
    })
    return payload


def role_gradient_audit(model: PACTOIAModel, batch: dict, device: torch.device, cfg: dict, license_value: float) -> dict:
    model.train(); model.zero_grad(set_to_none=True)
    batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    output = model(batch["image"], semantic_share_license=license_value,
                   action_scale=cfg["evidence"]["action_scale"], reason_budget=cfg["reason"]["selected_budget"])
    context = tuple(model.context_decoder.parameters()); explanation = tuple(model.explanation_decoder.parameters())
    shared = tuple(model.shared_readout.parameters())
    action_loss = asymmetric_loss_with_logits(output["action_logits_primary"], batch["action"])
    reason_loss = acpr_losses.partial_label_reason_loss(output["reason_logits_primary"], batch["reason"], output["contradiction_score"])

    def norms(loss, groups):
        gradients = torch.autograd.grad(loss, tuple(p for group in groups for p in group), retain_graph=True, allow_unused=True)
        result, offset = [], 0
        for group in groups:
            values = gradients[offset:offset + len(group)]; offset += len(group)
            result.append(float(torch.sqrt(sum((g.float().square().sum() for g in values if g is not None), start=loss.new_zeros(())))))
        return result

    action_context, action_explanation, action_shared = norms(action_loss, (context, explanation, shared))
    reason_context, reason_explanation, reason_shared = norms(reason_loss, (context, explanation, shared))
    model.zero_grad(set_to_none=True)
    return {"action_to_context": action_context, "action_to_explanation": action_explanation,
            "action_to_shared": action_shared, "reason_to_context": reason_context,
            "reason_to_explanation": reason_explanation, "reason_to_shared": reason_shared,
            "semantic_share_license": license_value,
            "illegal_cross_owner_grad_max": max(action_explanation, reason_context)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint", required=True); parser.add_argument("--mode", choices=("control", "pact"), required=True)
    parser.add_argument("--epochs", type=int); parser.add_argument("--batch-size", type=int); parser.add_argument("--num-workers", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--max-train-samples", type=int); parser.add_argument("--max-calib-samples", type=int); parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume")
    args = parser.parse_args(); cfg = load_config(args.config); out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(cfg.get("runtime", {}).get("cpu_threads", 4)))
    torch.set_num_interop_threads(1)
    if cfg["experiment"]["feature_cache_enabled"] or cfg["experiment"]["token_compression"] != "none":
        raise RuntimeError("PACT probe forbids feature cache and token compression")
    seed = int(cfg["data"]["split_seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device); torch.set_float32_matmul_precision("high")
    model = build_model(cfg, device); source = load_source(model, args.source_checkpoint)
    controller = PACTParetoController(cfg["pareto"]["candidates"], cfg["pareto"]["epsilon_action_audit"], cfg["pareto"]["license_ema"]).to(device)
    optimizer = build_optimizer(model, cfg); queue = PACTBalancedPairQueue(21, cfg["reason"]["queue_capacity"], cfg["reason"]["queue_capacity"], cfg["reason"]["queue_max_age_updates"])
    full_train, full_test = make_dataset(cfg, "train"), make_dataset(cfg, "test")
    all_names = [sample.file_name for sample in full_train.samples]
    split = stable_split_ids(all_names, seed, cfg["data"]["train_calib_fraction"], cfg["data"]["train_audit_count"])
    name_to_index = {sample.file_name: index for index, sample in enumerate(full_train.samples)}
    train_ids = list(range(len(full_train))); random.Random(seed).shuffle(train_ids)
    train_ids = train_ids[:args.max_train_samples or None]
    calib_ids = [name_to_index[name] for name in split["train_calib"][:args.max_calib_samples or None]]
    audit_ids = [name_to_index[name] for name in split["train_audit"]]
    test_ids = list(range(len(full_test)))[:args.max_test_samples or None]
    batch_size = args.batch_size or int(cfg["training"]["batch_size"]); workers = args.num_workers if args.num_workers is not None else int(cfg["data"]["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = make_loader(Subset(full_train, train_ids), batch_size, True, workers, cfg, generator)
    calib_loader = make_loader(Subset(full_train, calib_ids), batch_size, False, workers, cfg)
    audit_loader = make_loader(Subset(full_train, audit_ids), batch_size, False, workers, cfg)
    test_loader = make_loader(Subset(full_test, test_ids), batch_size, False, workers, cfg)
    split_manifest = out_dir / "split_manifest.json"
    write_split_manifest(split_manifest, all_names, seed, cfg["data"]["train_calib_fraction"], cfg["data"]["train_audit_count"])
    accumulation = args.gradient_accumulation_steps or int(cfg["training"]["gradient_accumulation_steps"])
    write_json(out_dir / "run_manifest.json", {
        "mode": args.mode, "source_checkpoint": str(Path(args.source_checkpoint).resolve()), **source,
        "source_head": cfg["experiment"]["source_head"],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command_line": [sys.executable, *sys.argv], "config_path": str(Path(args.config).resolve()),
        "config_hash": sha256(args.config), "checkpoint_hash": sha256(args.source_checkpoint),
        "split_hash": sha256(split_manifest), "split_seed": seed,
        "train_samples": len(train_ids), "calib_samples": len(calib_ids),
        "audit_samples": len(audit_ids), "test_samples": len(test_ids),
        "batch_size": batch_size, "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation, "num_workers": workers,
        "precision": cfg["training"]["precision"], "selected_layers": cfg["backbone"]["selected_layers"],
        "pretrained_weights": cfg["backbone"]["pretrained_weights"],
        "feature_cache_enabled": False, "token_compression": "none", "best_selection_split": "test",
        "loss_weights": cfg["loss_weights"],
    })
    structured_builder = AIEStructuredEvidenceBuilder(cfg["primary"]["scene_predicates"], cfg["data"]["bdd100k_root"])
    cf_fields = AIECounterfactualConfig.__dataclass_fields__
    cf_engine = AIECounterfactualEngine(AIECounterfactualConfig(**{key: cfg["counterfactual"][key] for key in cf_fields}))
    epochs = args.epochs or int(cfg["training"]["epochs"])
    total_updates = math.ceil(len(train_loader) / accumulation) * epochs; update = 0; start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); controller.load_state_dict(checkpoint["controller"])
        queue.load_state_dict(checkpoint["pair_queue"]); generator.set_state(checkpoint["generator_state"].cpu())
        update, start_epoch = checkpoint["update"], checkpoint["epoch"] + 1
    audit_iterator = iter(audit_loader)
    for epoch in range(start_epoch, epochs):
        model.train(); optimizer.zero_grad(set_to_none=True); micro = 0; start = time.perf_counter(); latest_stats = {}
        epoch_cf_cases = []; epoch_agreement = []; epoch_gate = []; epoch_named = []
        epoch_reason_bound = []; epoch_action_delta = []; epoch_contribution = []
        first_audit_batch = None; last_controller_result = None
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            structured_start = time.perf_counter()
            structured = structured_builder.build(batch["file_name"], device=device)
            structured_time = time.perf_counter() - structured_start
            if first_audit_batch is None:
                first_audit_batch = {key: (value.detach().cpu() if torch.is_tensor(value) else value) for key, value in batch.items()}
            license_value = 0.0 if args.mode == "control" else float(controller.semantic_share_license)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                field = model.encode_images(batch["image"])
                output = model.decode_from_field(
                    field, semantic_share_license=license_value,
                    action_scale=0.0 if args.mode == "control" else cfg["evidence"]["action_scale"],
                    reason_budget=0.0 if args.mode == "control" else cfg["reason"]["selected_budget"],
                    compatibility_mode=args.mode == "control")
                use_cf = counterfactual_due(
                    args.mode, update, micro, accumulation, int(cfg["counterfactual"]["interval_optimizer_updates"]))
                cf = cf_engine.run(model, output, batch["action"], batch["file_name"], global_update=update,
                                   action_scale=float(cfg["evidence"]["action_scale"])) if use_cf else None
                loss, rows, latest_stats = compute_losses(
                    output, batch, structured, cfg, queue, update, cf,
                    cf_accumulation_compensation=accumulation if use_cf else 1.0)
            if cf:
                epoch_cf_cases.extend(cf["cases"])
            epoch_agreement.append(float(output["predicate_visual_agreement"].mean().detach()))
            epoch_gate.append(float(output["predicate_agreement_strength"].mean().detach()))
            epoch_named.append(float((output["formal_predicate_name_id"] >= 0).float().mean().detach()))
            epoch_reason_bound.append(float(output["reason_delta_to_budget_max"].detach()))
            epoch_action_delta.extend(output["action_delta"].detach().float().cpu().flatten().tolist())
            epoch_contribution.extend(output["bounded_contribution"].detach().float().cpu().flatten().tolist())
            controller_result = None
            pareto_due = args.mode == "pact" and (update + 1) % int(cfg["pareto"]["update_interval"]) == 0 and micro + 1 == accumulation
            if pareto_due:
                try:
                    audit_batch = next(audit_iterator)
                except StopIteration:
                    audit_iterator = iter(audit_loader); audit_batch = next(audit_iterator)
                controller_result = pareto_controller_step(model, controller, output, batch, audit_batch, cfg, device)
            (loss / accumulation).backward(); micro += 1
            queue.enqueue(output["reason_logits_final_train"], batch["reason"], update, output["contradiction_score"])
            if micro == accumulation:
                multiplier = cosine_schedule(update, total_updates, cfg)
                for group in optimizer.param_groups: group["lr"] = group["base_lr"] * multiplier
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["global_grad_clip"])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1; micro = 0
                if controller_result is not None:
                    last_controller_result = controller_result
                    append_jsonl(out_dir / "pareto_license_stats.jsonl", controller_result)
                append_jsonl(out_dir / "loss_components.jsonl", {"epoch": epoch, "optimizer_update": update,
                             "total_loss": float(loss.detach()), "structured_time": structured_time, "terms": rows})
                if update == 1 or update % int(cfg["training"]["print_every_optimizer_updates"]) == 0:
                    print(json.dumps({"pact_batch": True, "mode": args.mode, "epoch": epoch, "update": update,
                                      "loss": float(loss.detach()), "structured_time": structured_time,
                                      "license": license_value,
                                      "reason_labels_with_pairs": latest_stats["labels_with_pairs"],
                                      "agreement": float(output.get("predicate_visual_agreement", torch.zeros(())).mean().detach()),
                                      "reason_bound_ratio": float(output["reason_delta_to_budget_max"].detach()) if args.mode == "pact" else 0.0}, sort_keys=True), flush=True)
        if micro:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["global_grad_clip"]); optimizer.step(); optimizer.zero_grad(set_to_none=True); update += 1
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        metrics = evaluate(model, calib_loader, test_loader, device, args.mode,
                           0.0 if args.mode == "control" else float(controller.semantic_share_license), cfg, epoch_dir)
        row = {"epoch": epoch, "seconds": time.perf_counter() - start, **metrics["deploy"]}
        append_jsonl(out_dir / "metrics_summary.jsonl", row); print(json.dumps({"pact_epoch": row}, sort_keys=True), flush=True)
        write_json(epoch_dir / "metrics_summary.json", row)
        write_json(epoch_dir / "pareto_license_stats.json", {"license": float(controller.semantic_share_license),
                   "last_update": last_controller_result})
        write_json(epoch_dir / "predicate_agreement_stats.json", {
            "agreement_mean": float(np.mean(epoch_agreement)), "gate_mean": float(np.mean(epoch_gate)),
            "agreement_p10": float(np.percentile(epoch_agreement, 10)), "agreement_p90": float(np.percentile(epoch_agreement, 90)),
            "gate_p10": float(np.percentile(epoch_gate, 10)), "gate_p90": float(np.percentile(epoch_gate, 90)),
            "named_coverage": float(np.mean(epoch_named)), "unnamed_visual_rate": 1.0 - float(np.mean(epoch_named))})
        reason_coverage = {key: value for key, value in latest_stats.items() if isinstance(value, (int, list))}
        reason_coverage.update({"reason_delta_to_budget_max": max(epoch_reason_bound),
                                "reason_delta_to_budget_mean": float(np.mean(epoch_reason_bound))})
        write_json(epoch_dir / "reason_rank_coverage.json", reason_coverage)
        action_rank = {key: value for key, value in latest_stats.items()
                       if key in ("primary_correct_pair_count", "final_preserved_pair_rate", "primary_wrong_pair_repair_rate", "new_pair_inversion_rate")}
        action_rank.update({"action_delta_rms": float(np.sqrt(np.mean(np.square(epoch_action_delta)))),
                            "action_delta_p10": float(np.percentile(epoch_action_delta, 10)),
                            "action_delta_p50": float(np.percentile(epoch_action_delta, 50)),
                            "action_delta_p90": float(np.percentile(epoch_action_delta, 90)),
                            "contribution_positive_rate": float(np.mean(np.asarray(epoch_contribution) > 0)),
                            "contribution_negative_rate": float(np.mean(np.asarray(epoch_contribution) < 0))})
        write_json(epoch_dir / "action_rank_stats.json", action_rank)
        cf_summary = counterfactual_case_metrics(epoch_cf_cases)
        if epoch_cf_cases:
            values = np.asarray([row["selected_minus_control"] for row in epoch_cf_cases], dtype=np.float64)
            rng = np.random.default_rng(20260809 + epoch)
            means = np.asarray([rng.choice(values, len(values), replace=True).mean() for _ in range(2000)])
            cf_summary["selected_minus_control_bootstrap_lcb"] = float(np.percentile(means, 2.5))
        write_json(epoch_dir / "counterfactual_summary.json", cf_summary)
        if first_audit_batch is not None:
            write_json(epoch_dir / "role_gradient_stats.json", role_gradient_audit(
                model, first_audit_batch, device, cfg, 0.0 if args.mode == "control" else float(controller.semantic_share_license)))
        checkpoint = {"model": canonical_model_state_dict(model.state_dict()), "optimizer": optimizer.state_dict(), "controller": controller.state_dict(),
                      "pair_queue": queue.state_dict(), "generator_state": generator.get_state(), "epoch": epoch, "update": update}
        torch.save(checkpoint, out_dir / "checkpoint_latest.pth")


if __name__ == "__main__":
    main()
