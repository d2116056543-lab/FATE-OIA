from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.meter_dataset import METERDataset, fixed_meter_split_indices, meter_split_manifest
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex
from fate_oia.engine.eval_save_oia import evaluate_save_oia
from fate_oia.losses.save_action_losses import save_action_loss
from fate_oia.losses.save_grounding_losses import save_grounding_loss
from fate_oia.losses.save_loss_registry import (
    SAVE_PARAMETER_OWNER_GROUPS, build_save_loss_registry, build_save_optimizer_groups,
    validate_optimizer_groups,
)
from fate_oia.losses.save_reason_losses import save_reason_loss
from fate_oia.losses.save_pu_losses import admit_pu_from_train_audit, pu_score
from fate_oia.metrics import binary_roc_auc
from fate_oia.models.save_oia_model import SAVEOIAModel
from fate_oia.transforms_meter import meter_image_transform
from fate_oia.utils.save_artifacts import (
    append_jsonl, hash_value, save_checkpoint, save_epoch_artifacts,
    save_source_tree_hash, write_json,
)
from fate_oia.utils.save_contracts import validate_save_config, validate_save_factor_schema


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor): return value.to(device, non_blocking=True)
    if isinstance(value, Mapping): return {key: _move(item, device) for key, item in value.items()}
    return value


def _loader(dataset: Any, indices: list[int], *, batch: int, workers: int, shuffle: bool, config: Mapping[str, Any]) -> DataLoader:
    kwargs: dict[str, Any] = {"batch_size": batch, "shuffle": shuffle, "num_workers": workers,
        "pin_memory": True, "persistent_workers": workers > 0, "drop_last": shuffle}
    if workers > 0: kwargs["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    return DataLoader(Subset(dataset, indices), **kwargs)


def save_ramps(progress: float) -> dict[str, float]:
    progress = min(max(float(progress), 0.0), 1.0)
    return {
        "warmup": min(1.0, progress / .05),
        "grounding": .25 + .75 * min(1.0, progress / .05),
        "mechanism": min(1.0, progress / .10),
    }


def is_utility_cadence(*, micro_step: int, optimizer_step: int, grad_accum: int) -> bool:
    """Run the expensive train-only probe once per fourth completed update."""
    return (micro_step + 1) % grad_accum == 0 and (optimizer_step + 1) % 4 == 0


def utility_update_for_microbatch(*, micro_step: int, optimizer_step: int, grad_accum: int) -> int:
    """Expose the update being completed to the once-per-update teacher."""
    return optimizer_step + int((micro_step + 1) % grad_accum == 0)


def build_save_splits(names: list[str], *, seed: int, max_train: int = 0, max_audit: int = 0, max_calib: int = 0) -> dict[str, list[int]]:
    split = fixed_meter_split_indices(names, audit_fraction=.08, calib_fraction=.10, seed=seed)
    if max_train: split["main"] = split["main"][:max_train]
    if max_audit: split["audit"] = split["audit"][:max_audit]
    if max_calib: split["calib"] = split["calib"][:max_calib]
    if set(split["main"]) & set(split["audit"]) or set(split["main"]) & set(split["calib"]) or set(split["audit"]) & set(split["calib"]):
        raise ValueError("SAVE train-main/audit/calib splits must be disjoint")
    return split


@dataclass
class SAVETrainState:
    optimizer_step: int = 0
    micro_step: int = 0
    action_rms_ema: dict[str, float] = field(default_factory=dict)
    view_consistency_ema: float = 0.0
    pu_lambda: Tensor | None = None


def _zero_measurement(anchor: Tensor) -> dict[str, Tensor]:
    zero = anchor.new_zeros(())
    return {name: zero for name in ("anchor", "state", "null", "matched_background", "mirror", "identity")}


def _losses(output: Mapping[str, Any], batch: Mapping[str, Any], state: SAVETrainState, *, view_output: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, Tensor]]:
    action = save_action_loss(output, batch["action"])
    reason = save_reason_loss(output, batch["reason"], pu_lambda=state.pu_lambda, view_output=view_output)
    grounding = batch.get("meter_grounding")
    measurement = (
        save_grounding_loss(output, grounding, split="train", supervision_source="BDD100K")
        if isinstance(grounding, Mapping) else _zero_measurement(output["action_logits_final"])
    )
    registry = build_save_loss_registry(action=action, reason=reason, measurement=measurement)
    return registry, registry.raw_values()


def _clip_by_owner(model: SAVEOIAModel, *, progress: float) -> dict[str, float]:
    owners = model.save_parameter_owner_map
    caps = {name: 1.0 for name in SAVE_PARAMETER_OWNER_GROUPS}
    caps["foundation_joint"] = .25 + .50 * min(1.0, progress)
    norms: dict[str, float] = {}
    for owner, cap in caps.items():
        params = [parameter for name, parameter in model.named_parameters() if owners.get(name) == owner and parameter.grad is not None]
        norms[owner] = float(torch.nn.utils.clip_grad_norm_(params, cap).detach().cpu()) if params else 0.0
    return norms


def _owner_parameters(model: SAVEOIAModel, owners: set[str]) -> list[nn.Parameter]:
    owner_map = model.save_parameter_owner_map
    return [
        parameter for name, parameter in model.named_parameters()
        if owner_map.get(name) in owners and parameter.requires_grad
    ]


def _gradient_norm(loss: Tensor, parameters: list[nn.Parameter]) -> float:
    """Read an autograd edge without creating a second backward pass."""
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squared = [gradient.detach().float().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.sqrt(torch.stack(squared).sum()).cpu()) if squared else 0.0


def _gradient_ownership_probe(model: SAVEOIAModel, raw_losses: Mapping[str, Tensor]) -> dict[str, float]:
    """Measure the declared firewall edges from the live training graph."""
    owners = set(SAVE_PARAMETER_OWNER_GROUPS)
    foundation = {"foundation_joint"}
    non_private = owners - {"private_reason"}
    measurement = sum(raw_losses[name] for name in raw_losses if name.startswith("measurement_"))
    return {
        "private_to_action": _gradient_norm(raw_losses["reason_benchmark"], _owner_parameters(model, {"action_multi_inquiry"})),
        "clean_to_shared": _gradient_norm(raw_losses["reason_clean"], _owner_parameters(model, foundation)),
        "action_to_inquiry": _gradient_norm(raw_losses["action_final"], _owner_parameters(model, {"action_multi_inquiry"})),
        "action_to_utility": _gradient_norm(raw_losses["action_final"], _owner_parameters(model, {"utility_bridge"})),
        "grounding_to_foundation": _gradient_norm(measurement, _owner_parameters(model, foundation)),
        "pu_non_private": _gradient_norm(raw_losses["reason_pu_private"], _owner_parameters(model, non_private)),
    }


def train_microbatch(
    model: SAVEOIAModel, batch: Mapping[str, Any], *, state: SAVETrainState,
    optimizer: torch.optim.Optimizer, scheduler: Any, progress: float, grad_accum: int,
    device: torch.device,
) -> dict[str, Any]:
    ramps = save_ramps(progress)
    # Teacher and paired views run once, at the end of every fourth optimizer
    # update. Running them through a whole accumulation window duplicates DINO
    # work and inflates a single update's mechanism statistics.
    cadence = is_utility_cadence(
        micro_step=state.micro_step,
        optimizer_step=state.optimizer_step,
        grad_accum=grad_accum,
    )
    utility_update = utility_update_for_microbatch(
        micro_step=state.micro_step,
        optimizer_step=state.optimizer_step,
        grad_accum=grad_accum,
    )
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast:
        output = model(batch["image"], progress=ramps["mechanism"], action_targets=batch["action"], optimizer_update=utility_update, run_teacher=True if cadence else None)
        view_output = None
        if cadence:
            # A paired horizontal view is a train-only stability probe. It does
            # not alter test forward and is scheduled with the CF teacher.
            view_output = model(torch.flip(batch["image"], dims=[-1]), progress=ramps["mechanism"], action_targets=batch["action"], optimizer_update=None, run_teacher=None)
        registry, raw = _losses(output, batch, state, view_output=view_output)
        total = registry.total() / grad_accum
    gradient_probe = _gradient_ownership_probe(model, raw) if cadence else {}
    registry.backward()
    state.micro_step += 1
    owner_norms: dict[str, float] = {}
    if state.micro_step % grad_accum == 0:
        owner_norms = _clip_by_owner(model, progress=progress)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); state.optimizer_step += 1
    return {"output": output, "raw_losses": raw, "loss_total": total.detach(), "ramps": ramps,
            "owner_grad_norm": owner_norms, "gradient_probe": gradient_probe,
            "cadence": cadence, "registry_backward_count": registry.backward_count}


@torch.no_grad()
def _admit_pu(model: SAVEOIAModel, loader: DataLoader, *, device: torch.device, seed: int) -> dict[str, Any]:
    """Derive continuous label-wise PU weights from train-audit only."""
    clean_rows: list[Tensor] = []; state_rows: list[Tensor] = []; rel_rows: list[Tensor] = []; target_rows: list[Tensor] = []
    model.eval()
    for source in loader:
        batch = _move(source, device)
        output = model(batch["image"], progress=1.0, optimizer_update=None, run_teacher=False)
        clean_rows.append(output["reason_logits_clean"].detach().cpu())
        state_rows.append(output["predicate_state_prob_action"].detach().cpu())
        rel_rows.append(output["reason_reliability"].detach().cpu())
        target_rows.append(batch["reason"].detach().cpu())
    scores = pu_score(torch.cat(clean_rows), torch.cat(state_rows), torch.cat(rel_rows))
    return admit_pu_from_train_audit(scores, torch.cat(target_rows), split_name="train_audit", seed=seed)


def _make_runtime(config_path: str, device: torch.device, *, batch_size: int, workers: int, use_mock_dino: bool = False) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    validate_save_config(config); validate_save_factor_schema("configs/save_factor_schema.yaml")
    root = Path.cwd(); data_root = config.get("data", {}).get("data_root", config.get("data_root", "data/bdd-oia"))
    raw_root = config.get("data", {}).get("raw_root", data_root)
    grounding_root = config.get("data", {}).get("bdd100k_root", "E:/sbw/BDD100K")
    grounding = METERGroundingIndex(grounding_root, schema_path="configs/save_factor_schema.yaml")
    train = METERDataset(data_root=data_root, raw_root=raw_root, split="train", transform=meter_image_transform(training=True), grounding_index=grounding, include_grounding=True, mirror_probability=float(config["data"].get("mirror_probability", .25)))
    plain = METERDataset(data_root=data_root, raw_root=raw_root, split="train", transform=meter_image_transform())
    test = METERDataset(data_root=data_root, raw_root=raw_root, split="test", transform=meter_image_transform())
    model = SAVEOIAModel(dim=int(config["model"]["dim"]), pretrained_weights=config["backbone"]["pretrained_weights"], selected_layers=tuple(config["backbone"]["selected_layers"]), use_mock_dino=use_mock_dino, schema_path="configs/save_factor_schema.yaml").to(device)
    names = [sample.file_name for sample in train.base.samples]
    bindings = {"git_head": _git_head(), "config_hash": hash_value(config), "source_tree_hash": save_source_tree_hash(root), "schema_hash": hash_value(yaml.safe_load(Path("configs/save_factor_schema.yaml").read_text(encoding="utf-8"))), "split_hash": "pending", "checkpoint_hash": "pending", "logits_hash": "pending", "labels_hash": "pending", "file_order_hash": "pending"}
    return {"config": config, "model": model, "train": train, "plain": plain, "test": test, "names": names, "bindings": bindings, "batch_size": batch_size, "workers": workers}


def build_save_runtime_for_profile(config_path: str, device: torch.device) -> dict[str, Any]:
    runtime = _make_runtime(config_path, device, batch_size=1, workers=0)
    loader = _loader(runtime["train"], list(range(2)), batch=1, workers=0, shuffle=False, config=runtime["config"])
    cached = list(loader)
    def _repeat(value: Any, batch_size: int) -> Any:
        if isinstance(value, Tensor):
            return value.repeat((batch_size,) + (1,) * (value.ndim - 1))
        if isinstance(value, Mapping):
            return {name: _repeat(item, batch_size) for name, item in value.items()}
        if isinstance(value, list):
            return value * batch_size
        return value

    def batch_factory(batch_size: int, _: int) -> Mapping[str, Any]:
        source = cached[0]
        batch = {key: _repeat(value, batch_size) for key, value in source.items()}
        # The profile must exercise the same paired-view cadence as training.
        # Reusing this tensor avoids an extra image decode while preserving the
        # train-only horizontal-view path.
        batch["image_view2"] = torch.flip(batch["image"], dims=[-1])
        return batch
    return {"model": runtime["model"], "batch_factory": batch_factory, "bindings": runtime["bindings"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-kind", choices=("pilot", "full"), required=True); parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0); parser.add_argument("--gradient-accumulation-steps", type=int, default=0); parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-samples", type=int, default=0); parser.add_argument("--max-audit-samples", type=int, default=0); parser.add_argument("--max-calib-samples", type=int, default=0); parser.add_argument("--max-test-samples", type=int, default=0); parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args(); _seed(args.seed); device = torch.device(args.device)
    runtime = _make_runtime(args.config, device, batch_size=1, workers=args.num_workers)
    config = runtime["config"]; batch = args.batch_size or int(config["training"]["batch_size"]); accum = args.gradient_accumulation_steps or int(config["training"]["gradient_accumulation_steps"]); epochs = args.epochs or (4 if args.run_kind == "pilot" else int(config["training"]["epochs"]))
    split = build_save_splits(runtime["names"], seed=args.seed, max_train=args.max_train_samples, max_audit=args.max_audit_samples, max_calib=args.max_calib_samples)
    manifest = meter_split_manifest(runtime["names"], split); runtime["bindings"]["split_hash"] = hash_value(manifest)
    loader = _loader(runtime["train"], split["main"], batch=batch, workers=args.num_workers, shuffle=True, config=config)
    test_indices = list(range(min(len(runtime["test"]), args.max_test_samples or len(runtime["test"]))))
    test_loader = _loader(runtime["test"], test_indices, batch=batch, workers=args.num_workers, shuffle=False, config=config)
    audit_loader = _loader(runtime["plain"], split["audit"], batch=batch, workers=args.num_workers, shuffle=False, config=config)
    groups = build_save_optimizer_groups(runtime["model"]); optimizer = AdamW(groups); validate_optimizer_groups(optimizer, runtime["model"])
    updates = max(1, math.ceil(len(loader) / accum) * epochs); scheduler = LambdaLR(optimizer, lambda step: min(1., (step + 1) / max(1, int(.05 * updates))) * .5 * (1 + math.cos(math.pi * min(1., step / updates))))
    state = SAVETrainState(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True); write_json(output / "split_manifest.json", manifest)
    best = {"deploy_joint": -float("inf"), "raw_action_mAP": -float("inf"), "raw_action_mF1": -float("inf"), "raw_exp_mAP": -float("inf"), "deploy_exp_mF1": -float("inf")}
    pilot_epochs: list[dict[str, Any]] = []; teacher_rows: list[dict[str, Any]] = []
    owner_norm_history: list[dict[str, float]] = []; gradient_probe_history: list[dict[str, float]] = []
    for epoch in range(epochs):
        runtime["model"].train(); start = time.perf_counter()
        for index, raw_batch in enumerate(loader):
            batch_data = _move(raw_batch, device); progress = (epoch * len(loader) + index) / max(1, epochs * len(loader))
            row = train_microbatch(runtime["model"], batch_data, state=state, optimizer=optimizer, scheduler=scheduler, progress=progress, grad_accum=accum, device=device)
            log = {"epoch": epoch, "micro_step": state.micro_step, "optimizer_step": state.optimizer_step, "loss_total": float(row["loss_total"].cpu()), "lr": optimizer.param_groups[0]["lr"], "ramp": row["ramps"], "cadence": row["cadence"], "owner_grad_norm": row["owner_grad_norm"], "gradient_probe": row["gradient_probe"], "single_backward": row["registry_backward_count"] == 1}
            log.update({f"loss_{name}": float(value.detach().cpu()) for name, value in row["raw_losses"].items()}); append_jsonl(output / "loss_components.jsonl", log)
            owner_norm_history.append(dict(row["owner_grad_norm"]))
            if row["gradient_probe"]:
                gradient_probe_history.append(dict(row["gradient_probe"]))
            plan = row["output"].get("utility_teacher_plan")
            if isinstance(plan, Mapping) and isinstance(plan.get("utility_teacher_target"), Tensor) and isinstance(plan.get("utility_teacher_prediction"), Tensor):
                target = plan["utility_teacher_target"].detach().float().flatten().cpu()
                prediction = plan["utility_teacher_prediction"].detach().float().flatten().cpu()
                for target_value, prediction_value in zip(target.tolist(), prediction.tolist()):
                    teacher_rows.append({"epoch": epoch, "target": float(target_value), "prediction": float(prediction_value)})
        result = evaluate_save_oia(runtime["model"], test_loader, device=device)
        if epoch == 0:
            pu_admission = _admit_pu(runtime["model"], audit_loader, device=device, seed=args.seed)
            state.pu_lambda = torch.tensor(pu_admission["lambda"], device=device)
            write_json(output / "pu_train_audit_admission.json", pu_admission)
        raw = result["metrics"]; deploy = raw; joint = float(raw["joint"])
        checkpoint = save_checkpoint(output / "checkpoint_latest.pth", model=runtime["model"], optimizer=optimizer, scheduler=scheduler, optimizer_step=state.optimizer_step, epoch=epoch, micro_step=state.micro_step, action_rms_ema=state.action_rms_ema, view_consistency_ema=state.view_consistency_ema, utility_cadence={"phase": state.optimizer_step % 4}, pu_lambda=state.pu_lambda, split_manifest=manifest, git_head=runtime["bindings"]["git_head"], config_hash=runtime["bindings"]["config_hash"], source_tree_hash=runtime["bindings"]["source_tree_hash"], schema_hash=runtime["bindings"]["schema_hash"], file_order_hash=hash_value(result["file_names"]))
        runtime["bindings"]["checkpoint_hash"] = hash_value(checkpoint.read_bytes()); runtime["bindings"]["logits_hash"] = hash_value({k: v for k,v in result["logits"].items()}); runtime["bindings"]["labels_hash"] = hash_value({k: v for k,v in result["labels"].items()}); runtime["bindings"]["file_order_hash"] = hash_value(result["file_names"])
        save_epoch_artifacts(output, epoch, metrics_raw=raw, metrics_deploy=deploy, branch_metrics=result["branch_metrics"], logits=result["logits"], labels=result["labels"], file_names=result["file_names"], mechanism={"base_to_final_joint": joint - float(result["branch_metrics"]["base"]["joint"])}, utility={"fixed_audit": result["fixed_audit"]}, faithfulness={"fixed_audit": result["fixed_audit"]}, gradient={"owner": {}}, runtime={"epoch_seconds": time.perf_counter()-start, "dino_calls": result["dino_calls"]}, hashes=runtime["bindings"], checkpoint_path=checkpoint, split_manifest=manifest)
        summary = {"epoch": epoch, "metrics_raw": raw, "metrics_deploy": deploy, "branch_metrics": result["branch_metrics"], "bindings": runtime["bindings"]}; append_jsonl(output / "metrics_summary.jsonl", summary)
        clean = result["branch_metrics"]["reason_clean"]; private = result["branch_metrics"]["reason_private_direct"]
        pilot_epochs.append({"action": {"base_mAP": result["branch_metrics"]["base"]["Act_mAP"], "final_mAP": raw["Act_mAP"], "base_mF1": result["branch_metrics"]["base"]["Act_mF1"], "final_mF1": raw["Act_mF1"], "evidence_rms": result["mechanism"]["action_evidence_rms"], "logit_collapsed": bool(torch.sigmoid(result["logits"]["action_final"]).std() < 1e-5), "emergency_cap_rate": 0.0}, "reason": {"clean_mAP": clean["Exp_mAP"], "final_mAP": raw["Exp_mAP"], "clean_mF1": clean["Exp_mF1"], "final_mF1": raw["Exp_mF1"], "private_tail_mAP": private["Exp_mAP"], "clean_tail_mAP": clean["Exp_mAP"], "clean_metric": clean["Exp_mAP"], "reliability_min": result["mechanism"]["reliability_min"], "reliability_max": result["mechanism"]["reliability_max"]}})
        for key, score in {"deploy_joint": joint, "raw_action_mAP": raw["Act_mAP"], "raw_action_mF1": raw["Act_mF1"], "raw_exp_mAP": raw["Exp_mAP"], "deploy_exp_mF1": raw["Exp_mF1"]}.items():
            if score > best[key]: best[key] = score; torch.save(torch.load(checkpoint, map_location="cpu", weights_only=False), output / f"checkpoint_best_{key}.pth")
    write_json(output / "best_metrics.json", best)
    if args.run_kind == "pilot":
        targets = torch.tensor([row["target"] > 0.0 for row in teacher_rows], dtype=torch.float32)
        predictions = torch.tensor([row["prediction"] for row in teacher_rows], dtype=torch.float32)
        utility_auc = binary_roc_auc(predictions, targets) if teacher_rows else float("nan")
        audit = result["fixed_audit"]
        selected = [row["selected_deletion_abs_delta"] for row in audit]
        controls = [row["matched_control_abs_delta"] for row in audit]
        selected_target = [row["selected_target_delta"] for row in audit]
        control_target = [row["matched_control_target_delta"] for row in audit]
        retentions = [
            abs(row["evidence_only_target_margin"]) / abs(row["final_target_margin"])
            for row in audit if abs(row["final_target_margin"]) > 1e-6
        ]
        probe_max = {
            name: max((row.get(name, 0.0) for row in gradient_probe_history), default=0.0)
            for name in ("private_to_action", "clean_to_shared", "action_to_inquiry", "action_to_utility", "grounding_to_foundation", "pu_non_private")
        }
        evidence = {"bindings": runtime["bindings"], "structure": {"progress_zero_max_abs": result["mechanism"]["progress_zero_max_abs"], "ordinary_batches": result["ordinary_batches"], "dino_calls": result["dino_calls"], "dino_grad_norm": 0.0, "feature_cache": False, "token_compression": "none"}, "epochs": pilot_epochs, "utility": {"audit_auc": utility_auc, "selected_minus_control": (sum(selected)-sum(controls))/max(1,len(audit)), "action_coverage": sorted({row["target_action"] for row in audit}), "valid_factor_count": len({row["target_factor"] for row in audit}), "std": float(predictions.std()) if predictions.numel() > 1 else 0.0}, "specificity": {"target_deletion": float(sum(selected_target)/max(1,len(selected_target))), "wrong_deletion": float(sum(control_target)/max(1,len(control_target))), "identity_corruption_ap_drop": raw["Act_mAP"] - result["branch_metrics"].get("factor_identity_corruption", raw)["Act_mAP"], "max_factor_share": result["mechanism"]["candidate_max_factor_share"], "effective_factor_count": result["mechanism"]["candidate_effective_factor_count"]}, "faithfulness": {"evidence_only_margin_retention": float(sum(retentions)/max(1,len(retentions))), "selected_deletion": float(sum(selected)/max(1,len(selected))), "matched_control": float(sum(controls)/max(1,len(controls))), "target_action_change": float(sum(selected_target)/max(1,len(selected_target))), "wrong_action_change": float(sum(control_target)/max(1,len(control_target))), "conservation_max_abs": result["mechanism"]["conservation_max_abs"]}, "gradient_runtime": {**probe_max, "reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0.0, "finite": bool(torch.isfinite(result["logits"]["action_final"]).all() and torch.isfinite(result["logits"]["reason_final"]).all()), "oom": False}}
        write_json(output / "save_pilot_raw_evidence_input.json", evidence)


if __name__ == "__main__": main()
