from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_task_aware_index import BDD100KTaskAwareIndex
from fate_oia.datasets.precise_grounding_adapter import PRECISEGroundingAdapter
from fate_oia.engine.eval_precise_oia import evaluate_precise
from fate_oia.losses.precise_losses import total_precise_losses
from fate_oia.models.precise_oia_model import PRECISEOIAModel
from fate_oia.transforms_precise import PRECISEImageTransform
from fate_oia.utils.precise_artifacts import append_jsonl, save_epoch_tensors, write_json, write_resolved_config
from fate_oia.utils.precise_gradient_ownership import ownership_snapshot, parameter_ownership
from fate_oia.utils.precise_runtime import gpu_memory_gb
from fate_oia.utils.precise_schema import load_evidence_fields
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def _config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": workers, "pin_memory": True}
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **kwargs)


def build_optimizers(model: PRECISEOIAModel, config: dict[str, Any]) -> list[torch.optim.Optimizer]:
    owner_config = config["optimizer"]
    mapped = {"reason_semantic": "reason_adapter_decoder", "exchange_reread": "exchange_and_reread"}
    optimizers = []
    for owner, parameters in parameter_ownership(model).items():
        setting = owner_config[mapped.get(owner, owner)]
        optimizers.append(torch.optim.AdamW(parameters, lr=float(setting["lr"]), weight_decay=float(setting["weight_decay"])))
    return optimizers


def build_schedulers(optimizers: list[torch.optim.Optimizer], updates_per_epoch: int, epochs: int, warmup_ratio: float) -> list[torch.optim.lr_scheduler.LambdaLR]:
    total = max(1, updates_per_epoch * epochs)
    warmup = max(1, int(total * warmup_ratio))
    def scale(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return [torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale) for optimizer in optimizers]


def _dataset_samples(dataset) -> list[Any]:
    return dataset.dataset.samples if isinstance(dataset, Subset) else dataset.samples


def build_train_grounding_targets(dataset, config: dict[str, Any], output_dir: Path) -> tuple[PRECISEGroundingAdapter, dict[str, dict[str, dict[str, Any]]]]:
    fields = load_evidence_fields(Path(config["evidence"]["field_config"]))
    adapter = PRECISEGroundingAdapter(fields)
    index = BDD100KTaskAwareIndex(config["bdd100k_root"], manifest_path=output_dir / "grounding_source_manifest.json")
    targets: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in _dataset_samples(dataset):
        targets[sample.file_name] = adapter.from_metadata(index.get(sample.file_name), index.metadata_for(sample.file_name))
    coverage = adapter.coverage(list(targets.values()))
    write_json(output_dir / "evidence_field_preflight.json", coverage)
    threshold = config["evidence"]
    unsupported = [name for name, value in coverage.items() if value["positive_count"] < threshold["min_positive"] or value["reliable_negative_count"] < threshold["min_reliable_negative"] or value["geometry_valid_count"] < threshold["min_geometry_valid"]]
    if unsupported:
        raise RuntimeError(f"PRECISE evidence fields failed coverage preflight: {unsupported}")
    return adapter, targets


def train(args: argparse.Namespace) -> None:
    config = _config(args.config)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = PRECISEImageTransform(return_mirror=False)
    train_set = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "train", 4, 21, True, transform)
    test_set = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "test", 4, 21, True, transform)
    if args.max_train_samples:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    if args.max_test_samples:
        test_set = Subset(test_set, range(min(args.max_test_samples, len(test_set))))
    main_indices, calib_indices = make_train_calib_indices(train_set, config["threshold"]["train_calib_fraction"])
    train_main = Subset(train_set, main_indices)
    train_calib = Subset(train_set, calib_indices)
    train_loader = _loader(train_main, args.batch_size, args.num_workers, True)
    calib_loader = _loader(train_calib, args.batch_size, args.num_workers, False)
    test_loader = _loader(test_set, args.batch_size, args.num_workers, False)
    model = PRECISEOIAModel(Path(args.config).parent, config["pretrained_weights"]).to(device)
    optimizers = build_optimizers(model, config)
    schedulers = build_schedulers(optimizers, math.ceil(len(train_loader) / args.gradient_accumulation_steps), args.epochs, config["training"]["warmup_ratio"])
    best = -float("inf")
    write_resolved_config(output_dir / "config_resolved.yaml", config)
    write_json(output_dir / "run_manifest.json", {"test_only": True, "best_selection_split": "test", "feature_cache_enabled": False, "token_compression": "none", "internal_test_selected": True, "publication_eligible_selection": False, "train_calib_count": len(calib_indices), "train_main_count": len(main_indices), "command_line": vars(args)})
    grounding_adapter, train_grounding = build_train_grounding_targets(train_set, config, output_dir)
    for epoch in range(args.epochs):
        model.train()
        last_output = None
        for micro_step, batch in enumerate(train_loader):
            output = model(batch["image"].to(device, non_blocking=True))
            target_batch = grounding_adapter.stack_batch([train_grounding[name] for name in batch["file_name"]], device)
            losses = total_precise_losses(output, batch["action"].to(device), batch["reason"].to(device), target_batch)
            deploy_loss = 0.02 * (torch.nn.functional.binary_cross_entropy_with_logits(output["action_logits_deploy"], batch["action"].to(device)) + torch.nn.functional.binary_cross_entropy_with_logits(output["reason_logits_deploy"], batch["reason"].to(device)))
            (losses["loss_total"] + deploy_loss).div(args.gradient_accumulation_steps).backward()
            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                for scheduler in schedulers:
                    scheduler.step()
            if micro_step % config["diagnostics"]["batch_log_every_optimizer_steps"] == 0:
                record = {key: float(value.detach().item()) for key, value in losses.items() if value.ndim == 0}
                record.update({"epoch": epoch, "micro_step": micro_step, "optimizer_step": (micro_step + 1) // args.gradient_accumulation_steps, "action_exchange_to_direct_ratio": float(output["action_exchange_delta"].pow(2).mean().sqrt().div(output["action_logits_direct"].pow(2).mean().sqrt().clamp_min(1e-6)).item()), "reason_exchange_to_direct_ratio": float(output["reason_exchange_delta"].pow(2).mean().sqrt().div(output["reason_logits_direct"].pow(2).mean().sqrt().clamp_min(1e-6)).item()), "explicit_reliability_mean": float(output["evidence_reliability"].mean().item()), "reference_center_collapse_rate": float(output["center_collapse_rate"].item()), **gpu_memory_gb(device), **ownership_snapshot(model)})
                append_jsonl(output_dir / "loss_components.jsonl", record)
            last_output = output
        # Never drop an incomplete accumulated tail batch.
        if len(train_loader) % args.gradient_accumulation_steps:
            for optimizer in optimizers:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
        # CalAlign is trained only on the deterministic train-calib split.
        model.threshold_head.train()
        for calib_batch in calib_loader:
            with torch.no_grad():
                raw = model(calib_batch["image"].to(device, non_blocking=True))
            threshold = model.threshold_head(raw["action_logits_final_raw"].detach(), raw["reason_logits_observed"].detach())
            threshold_loss = F.binary_cross_entropy_with_logits(threshold["action_logits_deploy"], calib_batch["action"].to(device)) + F.binary_cross_entropy_with_logits(threshold["reason_logits_deploy"], calib_batch["reason"].to(device))
            optimizers[-1].zero_grad(set_to_none=True); threshold_loss.backward(); optimizers[-1].step()
        model.threshold_head.update_teacher(model.threshold_head.compose_theta().detach(), ema=1.0)
        metrics, tensors = evaluate_precise(model, test_loader, device)
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        write_json(epoch_dir / "metrics_summary.json", metrics)
        write_json(epoch_dir / "branch_metrics.json", metrics)
        write_json(epoch_dir / "gradient_firewall.json", ownership_snapshot(model))
        if last_output is not None:
            write_json(epoch_dir / "evidence_family_stats.json", {"presence_mean": [float(value) for value in torch.sigmoid(last_output["evidence_presence_logits"]).mean(0).detach().cpu()], "observability_mean": [float(value) for value in torch.sigmoid(last_output["evidence_observability_logits"]).mean(0).detach().cpu()]})
            write_json(epoch_dir / "evidence_reliability.json", {"mean": [float(value) for value in last_output["evidence_reliability"].mean(0).detach().cpu()], "strong_rate": float((last_output["evidence_reliability"] >= 0.70).float().mean().item()), "weak_rate": float(((last_output["evidence_reliability"] >= 0.30) & (last_output["evidence_reliability"] < 0.70)).float().mean().item())})
            write_json(epoch_dir / "exchange_stats.json", {"overlap_mean": float(last_output["exchange_overlap"].mean().item()), "gate_mean": float(last_output["exchange_gate"].mean().item()), "gate_active_rate_gt_0p1": float((last_output["exchange_gate"] > 0.1).float().mean().item())})
            write_json(epoch_dir / "reread_stats.json", {"center_collapse_rate": float(last_output["center_collapse_rate"].item()), "out_of_bounds_rate": float(last_output["out_of_bounds_rate"].item()), "reference_variance": [float(value) for value in last_output["reference_point_variance"].detach().cpu()]})
            write_json(epoch_dir / "annotation_gap.json", {"annotation_delta_rms": float(last_output["annotation_delta"].pow(2).mean().sqrt().item()), "semantic_observed_gap_rms": float((last_output["reason_logits_semantic"] - last_output["reason_logits_observed"]).pow(2).mean().sqrt().item())})
        save_epoch_tensors(epoch_dir, {"action_logits_direct": tensors["action_direct"], "action_logits_final_raw": tensors["action_final_raw"], "action_logits_deploy": tensors["action_deploy"], "reason_logits_direct": tensors["reason_direct"], "reason_logits_semantic": tensors["reason_semantic"], "reason_logits_observed": tensors["reason_observed"], "reason_logits_deploy": tensors["reason_deploy"]}, tensors["labels_action"], tensors["labels_reason"])
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, "deploy_fixed_joint": metrics["deploy_fixed_joint"]})
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "optimizers": [item.state_dict() for item in optimizers], "schedulers": [item.state_dict() for item in schedulers], "rng_state": torch.get_rng_state(), "threshold_teacher": model.threshold_head.theta_teacher.detach().cpu(), "active_field_schema": [field["name"] for field in load_evidence_fields(config["evidence"]["field_config"])]}
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        if metrics["deploy_fixed_joint"] > best:
            best = metrics["deploy_fixed_joint"]
            torch.save(checkpoint, output_dir / "checkpoint_best_test_deploy_joint.pth")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
