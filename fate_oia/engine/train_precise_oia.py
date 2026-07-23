from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.bdd100k_task_aware_index import BDD100KTaskAwareIndex
from fate_oia.datasets.precise_grounding_adapter import PRECISEGroundingAdapter
from fate_oia.engine.eval_precise_oia import evaluate_precise
from fate_oia.engine.export_precise_cases import export_precise_cases
from fate_oia.engine.precise_curriculum import (
    curriculum_sha256,
    curriculum_state_for_epoch,
    owner_active_epoch_counts,
)
from fate_oia.losses.precise_losses import total_precise_losses
from fate_oia.losses.precise_losses import evidence_view_consistency_loss, refinement_loss, two_way_consistency_loss
from fate_oia.losses.precise_intervention_losses import (
    empty_packed_target_specific_interventions,
    packed_target_specific_interventions,
)
from fate_oia.models.precise_oia_model import PRECISEOIAModel
from fate_oia.models.precise_pcvl_probes import PRECISEPCVLProbes
from fate_oia.engine.run_precise_pcvl import evaluate_pcvl, train_pcvl_step
from fate_oia.transforms_precise import PRECISEImageTransform
from fate_oia.utils.precise_artifacts import append_jsonl, save_epoch_tensors, write_json, write_resolved_config
from fate_oia.utils.precise_gradient_ownership import grad_norm, ownership_snapshot, parameter_ownership, projected_target_credit_grads, tensor_list_norm
from fate_oia.utils.precise_runtime import gpu_memory_gb
from fate_oia.utils.precise_schema import load_evidence_fields
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def _config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _implementation_fingerprint(config_path: str | Path) -> dict[str, Any]:
    root = Path.cwd()
    sources = sorted(root.glob("fate_oia/**/*precise*.py"))
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode("utf-8")); digest.update(path.read_bytes())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        head = "unavailable"
    return {"git_head": head, "source_tree_sha256": digest.hexdigest(), "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(), "checked_source_count": len(sources)}


def verify_full_curriculum_authorization(
    config: dict[str, Any],
    fingerprint: dict[str, Any],
    gate_path: str | Path = ".review/PRECISE_OIA_V1_FULL_CURRICULUM_READY.json",
    require_clean: bool = True,
) -> dict[str, Any]:
    gate_file = Path(gate_path)
    if not gate_file.exists():
        raise RuntimeError("Full PRECISE training requires FULL_CURRICULUM_READY")
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    skill = Path(".codex/skills/precise-oia-implementation-audit/SKILL.md")
    expected = {
        "status": "FULL_CURRICULUM_READY",
        "git_head": fingerprint["git_head"],
        "config_sha256": fingerprint["config_sha256"],
        "training_source_sha256": fingerprint["source_tree_sha256"],
        "skill_sha256": hashlib.sha256(skill.read_bytes()).hexdigest(),
        "curriculum_sha256": curriculum_sha256(config),
        "override_source": "user_approved_2026-07-23",
    }
    mismatches = [name for name, value in expected.items() if gate.get(name) != value]
    if gate.get("unresolved"):
        mismatches.append("unresolved")
    if not gate.get("runtime_profile_passed"):
        mismatches.append("runtime_profile_passed")
    if not gate.get("curriculum_checks") or not all(gate["curriculum_checks"].values()):
        mismatches.append("curriculum_checks")
    if not gate.get("functional_checks") or not all(gate["functional_checks"].values()):
        mismatches.append("functional_checks")
    if require_clean and subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        mismatches.append("git_worktree_dirty")
    if mismatches:
        raise RuntimeError(f"FULL_CURRICULUM_READY identity/contract mismatch: {sorted(set(mismatches))}")
    return gate


def _index_sha(indices: list[int]) -> str:
    payload = ",".join(str(index) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _dataset_file_names(dataset) -> list[str]:
    if isinstance(dataset, Subset):
        parent = _dataset_file_names(dataset.dataset)
        return [parent[int(index)] for index in dataset.indices]
    return [str(sample.file_name) for sample in dataset.samples]


def _file_names_sha256(names: list[str]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()


def _observed_firewall(model: PRECISEOIAModel, observed_loss: torch.Tensor) -> dict[str, float]:
    owners = parameter_ownership(model)
    checked = (
        "action_foundation",
        "action_decoder",
        "reason_semantic",
        "evidence_core",
        "reread_adapter",
        "exchange_adapter",
        "reason_latent",
        "annotation_adapter",
    )
    parameters = [parameter for owner in checked for parameter in owners[owner]]
    gradients = torch.autograd.grad(observed_loss, parameters, retain_graph=True, allow_unused=True)
    result = {}
    offset = 0
    for owner in checked:
        owner_gradients = gradients[offset:offset + len(owners[owner])]
        offset += len(owners[owner])
        norms = [gradient.detach().norm() for gradient in owner_gradients if gradient is not None]
        result[f"observed_to_{owner}_grad_norm"] = float(torch.stack(norms).norm().item()) if norms else 0.0
    return result


def _loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": workers, "pin_memory": True}
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 4})
    return DataLoader(dataset, **kwargs)


def _training_split_indices(dataset, mode: str, seed: int, calib_fraction: float, train_trunk_on_all_train: bool) -> tuple[list[int], list[int], list[int]]:
    """Return main/audit/calibration indices with the plan-defined overlap rules."""
    if mode == "pilot":
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset), generator=generator).tolist()[: min(len(dataset), 4096 + 1024 + 512)]
        if len(order) < 4096 + 1024 + 512:
            raise RuntimeError("PRECISE pilot requires 5632 disjoint train rows")
        return order[:4096], order[4096:5120], order[5120:5632]
    split_main, calib = make_train_calib_indices(dataset, calib_fraction, seed=seed)
    main = list(range(len(dataset))) if train_trunk_on_all_train else split_main
    return main, [], calib


def build_optimizers(model: PRECISEOIAModel, config: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
    owner_config = config["optimizer"]
    optimizers = {}
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    for owner, parameters in parameter_ownership(model).items():
        setting = owner_config[owner]
        decay, no_decay = [], []
        for parameter in parameters:
            name = parameter_names[id(parameter)]
            if parameter.ndim <= 1 or name.endswith(".bias") or "embedding" in name or ".entity." in name or ".state." in name or ".sector." in name or ".role." in name:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": float(setting["weight_decay"])})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        optimizers[owner] = torch.optim.AdamW(groups, lr=float(setting["lr"]))
    return optimizers


def _step_owners(
    model: PRECISEOIAModel,
    owners: tuple[str, ...],
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    config: dict[str, Any],
    step_counts: dict[str, int],
    owner_active: dict[str, bool],
) -> dict[str, dict[str, float]]:
    parameters_by_owner = parameter_ownership(model)
    action_clip = float(config["training"]["grad_clip_action"])
    reason_clip = float(config["training"]["grad_clip_reason"])
    evidence_clip = float(config["training"]["grad_clip_evidence"])
    stats = {}
    for owner in owners:
        parameters = parameters_by_owner[owner]
        if not owner_active[owner]:
            optimizers[owner].zero_grad(set_to_none=True)
            stats[owner] = {
                "active": 0.0,
                "parameter_count": float(sum(parameter.numel() for parameter in parameters)),
                "grad_norm_pre_clip": 0.0,
                "grad_norm_post_clip": 0.0,
                "parameter_delta_norm": 0.0,
                "optimizer_step_count": float(step_counts[owner]),
                "lr": float(optimizers[owner].param_groups[0]["lr"]),
            }
            continue
        before = [parameter.detach().clone() for parameter in parameters]
        pre = float(grad_norm(parameters).item())
        clip = action_clip if owner.startswith("action") else evidence_clip if owner == "evidence_core" else reason_clip
        clip_grad_norm_(parameters, clip)
        post = float(grad_norm(parameters).item())
        optimizers[owner].step()
        delta = torch.stack([(parameter.detach() - old).norm() for parameter, old in zip(parameters, before)]).norm().item()
        optimizers[owner].zero_grad(set_to_none=True)
        schedulers[owner].step()
        step_counts[owner] += 1
        stats[owner] = {
            "active": 1.0,
            "parameter_count": float(sum(parameter.numel() for parameter in parameters)),
            "grad_norm_pre_clip": pre,
            "grad_norm_post_clip": post,
            "parameter_delta_norm": float(delta),
            "optimizer_step_count": float(step_counts[owner]),
            "lr": float(optimizers[owner].param_groups[0]["lr"]),
        }
    return stats


def build_schedulers(optimizers: dict[str, torch.optim.Optimizer], updates_per_epoch: int | dict[str, int], active_epochs: int | dict[str, int], warmup_ratio: float) -> dict[str, torch.optim.lr_scheduler.LambdaLR]:
    schedulers = {}
    for owner, optimizer in optimizers.items():
        owner_updates = updates_per_epoch[owner] if isinstance(updates_per_epoch, dict) else updates_per_epoch
        owner_epochs = active_epochs[owner] if isinstance(active_epochs, dict) else active_epochs
        total = max(1, owner_updates * owner_epochs)
        warmup = max(1, int(total * warmup_ratio))
        def scale(step: int, total_steps: int = total, warmup_steps: int = warmup) -> float:
            if step < warmup_steps:
                return float(step + 1) / warmup_steps
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        schedulers[owner] = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)
    return schedulers


def load_resume_checkpoint(
    checkpoint_path: str | Path,
    model: PRECISEOIAModel,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    expected_fingerprint: dict[str, Any],
    expected_curriculum_sha256: str,
    expected_owner_active_epochs: dict[str, int],
    curriculum_config: dict[str, Any] | None = None,
    updates_per_epoch: dict[str, int] | None = None,
    pcvl_probes: PRECISEPCVLProbes | None = None,
    pcvl_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int, int, float, dict[str, float], dict[str, int], int, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("implementation_fingerprint") != expected_fingerprint:
        raise RuntimeError("Resume checkpoint implementation/config fingerprint does not match current code")
    if checkpoint.get("curriculum_sha256") != expected_curriculum_sha256:
        raise RuntimeError("Resume checkpoint curriculum hash does not match current schedule")
    if checkpoint.get("owner_active_epochs") != expected_owner_active_epochs:
        raise RuntimeError("Resume checkpoint owner active-epoch totals do not match current schedule")
    required_lifecycle = {
        "curriculum_state",
        "optimizer_step_counts",
        "optimizers",
        "schedulers",
        "owner_step_deltas",
    }
    missing_lifecycle = sorted(required_lifecycle - set(checkpoint))
    if missing_lifecycle:
        raise RuntimeError(f"Resume checkpoint is missing lifecycle fields: {missing_lifecycle}")
    if curriculum_config is not None:
        if updates_per_epoch is None:
            raise RuntimeError("Strict curriculum resume requires updates_per_epoch")
        saved_epoch = int(checkpoint["epoch"])
        expected_state = curriculum_state_for_epoch(curriculum_config, saved_epoch).to_dict()
        expected_state.pop("owner_active")
        if checkpoint["curriculum_state"] != expected_state:
            raise RuntimeError("Resume checkpoint curriculum state does not match its saved epoch")
        expected_steps = {
            owner: int(updates_per_epoch[owner])
            * sum(
                int(curriculum_state_for_epoch(curriculum_config, epoch).owner_active[owner])
                for epoch in range(saved_epoch + 1)
            )
            for owner in expected_owner_active_epochs
        }
        saved_steps = {owner: int(value) for owner, value in checkpoint["optimizer_step_counts"].items()}
        if saved_steps != expected_steps:
            raise RuntimeError("Resume checkpoint owner-local step counters are inconsistent")
        for owner in expected_owner_active_epochs:
            optimizer_state = checkpoint["optimizers"][owner]["state"]
            if bool(optimizer_state) != (expected_steps[owner] > 0):
                raise RuntimeError(f"Resume checkpoint optimizer lifecycle mismatch for {owner}")
            if int(checkpoint["schedulers"][owner]["last_epoch"]) != expected_steps[owner]:
                raise RuntimeError(f"Resume checkpoint scheduler lifecycle mismatch for {owner}")
    current_fields = [field["name"] for field in model.evidence_schema]
    if checkpoint.get("active_field_schema") != current_fields:
        raise RuntimeError("Resume checkpoint active evidence schema does not match current preflight")
    model.load_state_dict(checkpoint["model"])
    for owner, optimizer in optimizers.items():
        optimizer.load_state_dict(checkpoint["optimizers"][owner])
    for owner, scheduler in schedulers.items():
        scheduler.load_state_dict(checkpoint["schedulers"][owner])
    torch.set_rng_state(checkpoint["rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    if pcvl_probes is not None:
        if "pcvl_probes" not in checkpoint or "pcvl_optimizer" not in checkpoint:
            raise RuntimeError("Pilot resume checkpoint is missing PCVL probe state")
        pcvl_probes.load_state_dict(checkpoint["pcvl_probes"])
        if pcvl_optimizer is None:
            raise RuntimeError("Pilot resume requires a PCVL optimizer")
        pcvl_optimizer.load_state_dict(checkpoint["pcvl_optimizer"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["global_optimizer_step"]), int(checkpoint.get("global_micro_step", 0)), float(checkpoint.get("best_deploy_joint", -float("inf"))), dict(checkpoint.get("best_scores", {})), {key: int(value) for key, value in checkpoint.get("optimizer_step_counts", {}).items()}, int(checkpoint.get("pcvl_optimizer_step_count", 0)), int(checkpoint.get("pcvl_nonzero_update_count", 0))


def _dataset_samples(dataset) -> list[Any]:
    if not isinstance(dataset, Subset):
        return dataset.samples
    parent = _dataset_samples(dataset.dataset)
    return [parent[int(index)] for index in dataset.indices]


_STATIC_OUTPUT_TENSOR_KEYS = {
    "evidence_view_consistency",
    "action_evidence_family_mask",
    "evidence_part_valid",
    "evidence_state_channel_valid",
    "evidence_geometry_type",
}


def _slice_batch_output(value: Any, stop: int, full_batch: int, start: int = 0, key: str | None = None) -> Any:
    """Slice batch-valued model outputs without corrupting schema tensors.

    Static evidence tensors can have a leading dimension equal to a runtime
    batch size by coincidence, so shape alone is not a sufficient contract.
    """
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == full_batch and key not in _STATIC_OUTPUT_TENSOR_KEYS:
        return value[start:stop]
    if isinstance(value, dict):
        return {child_key: _slice_batch_output(item, stop, full_batch, start=start, key=child_key) for child_key, item in value.items()}
    return value


def select_active_evidence_fields(fields: list[dict[str, Any]], coverage: dict[str, dict[str, int]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    unsupported = {
        field["name"] for field in fields
        if coverage[field["name"]]["positive_count"] < config["min_positive"]
        or coverage[field["name"]]["reliable_negative_count"] < config["min_reliable_negative"]
        or (field.get("geometry_required", False) and coverage[field["name"]]["geometry_valid_count"] < config["min_geometry_valid"])
    }
    mirror = {"actor_left": "actor_right", "actor_right": "actor_left", "drivable_left": "drivable_right", "drivable_right": "drivable_left", "boundary_left": "boundary_right", "boundary_right": "boundary_left"}
    unsupported |= {mirror[name] for name in list(unsupported) if name in mirror}
    active = [field for field in fields if field["name"] not in unsupported]
    if not active:
        raise RuntimeError("No explicit PRECISE evidence field passed coverage preflight")
    return active, sorted(unsupported)


def build_train_grounding_targets(dataset, config: dict[str, Any], output_dir: Path, enforce_coverage: bool = True) -> tuple[PRECISEGroundingAdapter, dict[str, dict[str, dict[str, Any]]], list[dict[str, Any]]]:
    fields = load_evidence_fields(Path(config["evidence"]["field_config"]))
    adapter = PRECISEGroundingAdapter(fields)
    index = BDD100KTaskAwareIndex(config["bdd100k_root"], manifest_path=output_dir / "grounding_source_manifest.json")
    targets: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in _dataset_samples(dataset):
        targets[sample.file_name] = adapter.from_metadata(index.get(sample.file_name), index.metadata_for(sample.file_name))
    coverage = adapter.coverage(list(targets.values()))
    write_json(output_dir / "evidence_field_preflight.json", coverage)
    active_fields, unsupported = select_active_evidence_fields(fields, coverage, config["evidence"]) if enforce_coverage else (fields, [])
    active_names = {field["name"] for field in active_fields}
    active_targets = {sample: {name: value for name, value in sample_targets.items() if name in active_names} for sample, sample_targets in targets.items()}
    active_adapter = PRECISEGroundingAdapter(active_fields)
    with (output_dir / "active_evidence_fields.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"explicit_fields": active_fields, "disabled_fields": unsupported}, handle, sort_keys=False)
    return active_adapter, active_targets, active_fields


def train(args: argparse.Namespace) -> None:
    config = _config(args.config)
    if args.mode == "full" and not args.allow_full_with_embedded_curriculum:
        raise RuntimeError("Full PRECISE training requires the approved embedded-curriculum override")
    if args.mode == "full" and args.epochs != int(config["curriculum"]["epochs"]):
        raise RuntimeError("Full PRECISE epochs must match the fixed curriculum")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _implementation_fingerprint(args.config)
    if args.mode == "full":
        verify_full_curriculum_authorization(config, fingerprint)
    augmentation = config["augmentation"]
    train_transform = PRECISEImageTransform(return_mirror=False, training=True, brightness=float(augmentation["brightness"]), contrast=float(augmentation["contrast"]))
    eval_transform = PRECISEImageTransform(return_mirror=False, training=False)
    full_train_set = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "train", 4, 21, True, train_transform)
    full_train_eval_set = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "train", 4, 21, True, eval_transform)
    train_set = full_train_set
    train_eval_set = full_train_eval_set
    test_set = BDDOIAMultiTaskDataset(config["data_root"], config["raw_root"], "test", 4, 21, True, eval_transform)
    if args.max_train_samples:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
        train_eval_set = Subset(train_eval_set, range(min(args.max_train_samples, len(train_eval_set))))
    if args.max_test_samples:
        test_set = Subset(test_set, range(min(args.max_test_samples, len(test_set))))
    main_indices, audit_indices, calib_indices = _training_split_indices(
        train_set,
        args.mode,
        args.seed,
        float(config["threshold"]["train_calib_fraction"]),
        bool(config["threshold"].get("train_trunk_on_all_train", False)),
    )
    train_main = Subset(train_set, main_indices)
    train_calib = Subset(train_eval_set, calib_indices)
    train_loader = _loader(train_main, args.batch_size, args.num_workers, True)
    calib_loader = _loader(train_calib, args.batch_size, args.num_workers, False)
    audit_loader = _loader(Subset(train_eval_set, audit_indices), args.batch_size, args.num_workers, False) if audit_indices else None
    test_loader = _loader(test_set, args.batch_size, args.num_workers, False)
    # Field eligibility is a dataset property, not a smoke-subsample property.
    grounding_adapter, train_grounding, active_fields = build_train_grounding_targets(full_train_set, config, output_dir)
    model = PRECISEOIAModel(Path(args.config).parent, config["pretrained_weights"], evidence_schema=active_fields, model_config=config).to(device)
    optimizers = build_optimizers(model, config)
    trunk_updates = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    updates_by_owner = {owner: (len(calib_loader) if owner == "threshold_head" else trunk_updates) for owner in optimizers}
    active_epochs_by_owner = owner_active_epoch_counts(config, args.epochs) if args.mode == "full" else {owner: args.epochs for owner in optimizers}
    curriculum_digest = curriculum_sha256(config)
    schedulers = build_schedulers(optimizers, updates_by_owner, active_epochs_by_owner, config["training"]["warmup_ratio"])
    best = -float("inf")
    best_scores = {"action_mf1": -float("inf"), "exp_mf1": -float("inf"), "exp_map": -float("inf"), "semantic_exp_map": -float("inf")}
    start_epoch = 0
    global_optimizer_step = 0
    global_micro_step = 0
    resumed_step_counts: dict[str, int] = {}
    pcvl_probes = PRECISEPCVLProbes().to(device) if args.mode == "pilot" else None
    pcvl_optimizer = torch.optim.AdamW(pcvl_probes.parameters(), lr=3e-4) if pcvl_probes is not None else None
    pcvl_optimizer_step_count = 0
    pcvl_nonzero_update_count = 0
    last_pcvl_update = {"loss": 0.0, "grad_norm": 0.0, "parameter_delta_norm": 0.0}
    if args.resume_checkpoint:
        start_epoch, global_optimizer_step, global_micro_step, best, resumed_best_scores, resumed_step_counts, pcvl_optimizer_step_count, pcvl_nonzero_update_count = load_resume_checkpoint(
            args.resume_checkpoint,
            model,
            optimizers,
            schedulers,
            device,
            fingerprint,
            curriculum_digest,
            active_epochs_by_owner,
            config,
            updates_by_owner,
            pcvl_probes,
            pcvl_optimizer,
        )
        best_scores.update(resumed_best_scores)
    write_resolved_config(output_dir / "config_resolved.yaml", config)
    write_json(output_dir / "implementation_fingerprint.json", fingerprint)
    skill_path = Path(".codex/skills/precise-oia-implementation-audit/SKILL.md")
    action_schema_path = Path(args.config).parent / "precise_action_semantics.yaml"
    audit_file_names = _dataset_file_names(Subset(train_eval_set, audit_indices)) if audit_indices else []
    test_file_names = _dataset_file_names(test_set)
    run_manifest = {"git_head": fingerprint["git_head"], "source_tree_sha256": fingerprint["source_tree_sha256"], "config_sha256": fingerprint["config_sha256"], "skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest() if skill_path.exists() else "missing", "action_schema_sha256": hashlib.sha256(action_schema_path.read_bytes()).hexdigest(), "base_commit": "373aa49feac17372574fd7fb056c1d79c7c848fe", "test_only": True, "selection_protocol": "internal_test_selected", "best_selection_split": "test", "feature_cache_enabled": False, "token_compression": "none", "internal_test_selected": True, "publication_eligible_selection": False, "selected_layers": [3, 7, 11], "pretrained_weights": config["pretrained_weights"], "pretrained_weights_sha256": model.dino.pretrained_weights_sha256, "dino_loaded_state_key_count": len(model.dino.loaded_state_keys), "dino_missing_keys": list(model.dino.missing_keys), "dino_unexpected_keys": list(model.dino.unexpected_keys), "seed": args.seed, "epochs": args.epochs, "train_dataset_count": len(train_set), "test_count": len(test_set), "test_file_names_sha256": _file_names_sha256(test_file_names), "train_calib_count": len(calib_indices), "train_main_count": len(main_indices), "train_audit_count": len(audit_indices), "train_main_indices_sha256": _index_sha(main_indices), "train_audit_indices_sha256": _index_sha(audit_indices), "train_audit_file_names_sha256": _file_names_sha256(audit_file_names), "train_calib_indices_sha256": _index_sha(calib_indices), "train_trunk_on_all_train": bool(config["threshold"].get("train_trunk_on_all_train", False)), "pcvl_pilot_only": args.mode == "pilot", "active_evidence_fields": [field["name"] for field in active_fields], "command_line": vars(args), "curriculum_name": config["curriculum"]["name"], "curriculum_version": config["curriculum"]["version"], "curriculum_schedule": config["curriculum"]["schedule"], "curriculum_sha256": curriculum_digest, "owner_active_epochs": active_epochs_by_owner, "embedded_curriculum_override": bool(args.allow_full_with_embedded_curriculum), "override_source": "user_approved_2026-07-23" if args.allow_full_with_embedded_curriculum else None}
    write_json(output_dir / "run_manifest.json", run_manifest)
    profile_path = Path(".review/precise_oia_v1/runtime/selected_runtime_profile.json")
    write_json(output_dir / "runtime_profile.json", json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {"available": False, "reason": "actual-path profile has not passed yet"})
    representation_owners = tuple(owner for owner in optimizers if owner != "threshold_head")
    optimizer_step_counts = {owner: resumed_step_counts.get(owner, 0) for owner in optimizers}
    last_owner_stats: dict[str, dict[str, float]] = {}
    autocast_enabled = device.type == "cuda" and config["training"]["precision"] == "bf16"
    last_batch_finished = time.perf_counter()
    firewall_report: dict[str, float] = {}
    last_threshold_loss = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs):
        curriculum_state = curriculum_state_for_epoch(config, epoch) if args.mode == "full" else None
        if curriculum_state is not None:
            model.set_curriculum_state(curriculum_state)
            owner_active = curriculum_state.owner_active
        else:
            owner_active = {owner: True for owner in optimizers}
        epoch_start_step_counts = dict(optimizer_step_counts)
        model.train()
        last_output = None
        for micro_step, batch in enumerate(train_loader):
            global_micro_step += 1
            batch_started = time.perf_counter()
            data_time = batch_started - last_batch_finished
            images = batch["image"].to(device, non_blocking=True)
            batch_size = images.shape[0]
            mirror_count = min(batch_size, max(0, int(round(batch_size * float(config["augmentation"]["mirror_pair_fraction"])))))
            model_input = torch.cat([images, images[:mirror_count].flip(-1)], dim=0) if mirror_count else images
            dino_started = time.perf_counter()
            dino_calls_before = int(model.dino.dino_call_count)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                dino_output = model.dino(model_input)
            dino_call_count_batch = int(model.dino.dino_call_count) - dino_calls_before
            dino_time = time.perf_counter() - dino_started
            visual_started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                visual_field = model.visual_field(dino_output)
            visual_field_time = time.perf_counter() - visual_started
            decode_started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                full_output = model.decode_from_field(visual_field)
            decode_time = time.perf_counter() - decode_started
            output = _slice_batch_output(full_output, batch_size, model_input.shape[0])
            mirror_output = None
            if mirror_count:
                mirror_output = _slice_batch_output(full_output, model_input.shape[0], model_input.shape[0], start=batch_size)
            target_batch = grounding_adapter.stack_batch([train_grounding[name] for name in batch["file_name"]], device)
            action_target = batch["action"].to(device)
            reason_target = batch["reason"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                losses = total_precise_losses(output, action_target, reason_target, target_batch)
            if pcvl_probes is not None and pcvl_optimizer is not None:
                last_pcvl_update = train_pcvl_step(pcvl_probes, pcvl_optimizer, output, target_batch, action_target)
                pcvl_optimizer_step_count += 1
                pcvl_nonzero_update_count += int(last_pcvl_update["parameter_delta_norm"] > 0.0 and last_pcvl_update["grad_norm"] > 0.0)
                losses["loss_pcvl_detached"] = torch.tensor(last_pcvl_update["loss"], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                loss_refinement = 0.05 * (
                    refinement_loss(output["action_logits_direct"], output["action_logits_final_raw"], action_target)
                    + refinement_loss(output["reason_logits_direct"], output["reason_logits_semantic"], reason_target)
                )
            loss_mirror = losses["loss_total"] * 0.0
            loss_evidence_view = losses["loss_total"] * 0.0
            if mirror_output is not None:
                action_mirror = torch.tensor([0, 1, 3, 2], device=device)
                reason_mirror = torch.tensor([int(row["mirror_partner"]) for row in model.reason_schema], device=device)
                field_mirror = model.evidence_fields.mirror_field_indices
                loss_evidence_view = evidence_view_consistency_loss(output, mirror_output, field_mirror)
                mirror_composite = (
                    two_way_consistency_loss(output["action_logits_final_raw"][:mirror_count], mirror_output["action_logits_final_raw"], action_mirror)
                    + two_way_consistency_loss(output["reason_logits_semantic"][:mirror_count], mirror_output["reason_logits_semantic"], reason_mirror)
                    + loss_evidence_view
                )
                loss_mirror = 0.02 * mirror_composite
                model.evidence_fields.update_view_consistency(output["explicit_evidence_tokens"][:mirror_count].detach(), mirror_output["explicit_evidence_tokens"].detach())
            intervention_scale = float(model.curriculum_state()["intervention"])
            if intervention_scale > 0.0:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    intervention = packed_target_specific_interventions(
                        model,
                        output,
                        action_target,
                        reason_target,
                        int(config["intervention"]["max_pairs_per_batch"]),
                    )
            else:
                intervention = empty_packed_target_specific_interventions(
                    output["action_logits_final_raw"],
                    output["reason_logits_semantic"],
                )
            intervention_raw_loss = intervention["loss_intervention"]
            intervention["loss_intervention_raw"] = intervention_raw_loss
            intervention["loss_intervention"] = intervention_scale * intervention_raw_loss
            intervention["intervention_activation"] = torch.tensor(intervention_scale, device=device)
            total_updates = max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs)
            auxiliary_warmup = min(1.0, float(global_optimizer_step + 1) / max(1.0, 0.10 * total_updates))
            latent_regularizer = 0.15 * 0.02 * losses["loss_evidence_latent_diversity"]
            main_loss = losses["loss_total"] - 0.15 * losses["loss_evidence"] - latent_regularizer
            total_loss = main_loss + auxiliary_warmup * (
                0.15 * losses["loss_evidence"]
                + latent_regularizer
                + loss_refinement
                + loss_mirror
                + intervention["loss_intervention"]
            )
            evidence_parameters = parameter_ownership(model)["evidence_core"]
            grounding_credit, raw_credit, projected_credit = projected_target_credit_grads(
                auxiliary_warmup * 0.15 * losses["loss_evidence"],
                auxiliary_warmup * intervention["loss_intervention"],
                evidence_parameters,
                float(config["intervention"]["target_credit_grad_ratio"]),
            )
            if micro_step == 0:
                firewall_report = _observed_firewall(model, losses["loss_reason_observed"])
            backward_started = time.perf_counter()
            total_loss.div(args.gradient_accumulation_steps).backward()
            for parameter, raw_grad, projected_grad in zip(evidence_parameters, raw_credit, projected_credit):
                if parameter.grad is not None and raw_grad is not None and projected_grad is not None:
                    parameter.grad.add_((projected_grad - raw_grad) / args.gradient_accumulation_steps)
            backward_time = time.perf_counter() - backward_started
            losses.update({"loss_total": total_loss, "loss_refinement": loss_refinement, "loss_mirror": loss_mirror, "loss_evidence_view": loss_evidence_view, **intervention, "auxiliary_warmup": torch.tensor(auxiliary_warmup, device=device)})
            did_step = (micro_step + 1) % args.gradient_accumulation_steps == 0
            optimizer_time = 0.0
            if did_step:
                optimizer_started = time.perf_counter()
                last_owner_stats = _step_owners(model, representation_owners, optimizers, schedulers, config, optimizer_step_counts, owner_active)
                optimizer_time = time.perf_counter() - optimizer_started
                global_optimizer_step += 1
            if did_step and global_optimizer_step % int(config["diagnostics"]["batch_log_every_optimizer_steps"]) == 0:
                record = {key: float(value.detach().item()) for key, value in losses.items() if value.ndim == 0}
                elapsed = max(time.perf_counter() - batch_started, 1e-8)
                action_direct_error = F.binary_cross_entropy_with_logits(output["action_logits_direct"].detach(), action_target.float(), reduction="none").mean(-1)
                action_refined_error = F.binary_cross_entropy_with_logits(output["action_logits_final_raw"].detach(), action_target.float(), reduction="none").mean(-1)
                reason_direct_error = F.binary_cross_entropy_with_logits(output["reason_logits_direct"].detach(), reason_target.float(), reduction="none").mean(-1)
                reason_refined_error = F.binary_cross_entropy_with_logits(output["reason_logits_semantic"].detach(), reason_target.float(), reduction="none").mean(-1)
                action_hard = action_direct_error >= torch.quantile(action_direct_error.float(), 0.5).to(action_direct_error)
                reason_hard = reason_direct_error >= torch.quantile(reason_direct_error.float(), 0.5).to(reason_direct_error)
                hard_improvement = 0.5 * (
                    ((action_refined_error < action_direct_error) & action_hard).float().sum() / action_hard.float().sum().clamp_min(1.0)
                    + ((reason_refined_error < reason_direct_error) & reason_hard).float().sum() / reason_hard.float().sum().clamp_min(1.0)
                )
                easy_regression = 0.5 * (
                    ((action_refined_error > action_direct_error + 0.02) & ~action_hard).float().sum() / (~action_hard).float().sum().clamp_min(1.0)
                    + ((reason_refined_error > reason_direct_error + 0.02) & ~reason_hard).float().sum() / (~reason_hard).float().sum().clamp_min(1.0)
                )
                action_direct_rms = output["action_logits_direct"].pow(2).mean().sqrt().clamp_min(1e-6)
                reason_direct_rms = output["reason_logits_direct"].pow(2).mean().sqrt().clamp_min(1e-6)
                record.update({"event": "precise_batch", "epoch": epoch, "micro_step": micro_step, "optimizer_step": global_optimizer_step, "data_time": data_time, "dino_time": dino_time, "visual_field_time": visual_field_time, "evidence_time": float(output["diagnostics"]["evidence_seconds"]), "decode_time": decode_time, "backward_time": backward_time, "optimizer_time": optimizer_time, "samples_per_sec": float(batch_size / elapsed), "loss_threshold": last_threshold_loss, "action_direct_logit_rms": float(action_direct_rms), "action_reread_delta_rms": float(output["action_reread_delta_effective"].pow(2).mean().sqrt()), "action_reread_delta_raw_rms": float(output["action_reread_delta_raw"].pow(2).mean().sqrt()), "action_exchange_delta_rms": float(output["action_exchange_delta_effective"].pow(2).mean().sqrt()), "action_exchange_delta_raw_rms": float(output["action_exchange_delta_raw"].pow(2).mean().sqrt()), "action_reread_to_direct_ratio": float(output["action_reread_delta_effective"].pow(2).mean().sqrt().div(action_direct_rms).item()), "action_exchange_to_direct_ratio": float(output["action_exchange_delta_effective"].pow(2).mean().sqrt().div(action_direct_rms).item()), "reason_direct_logit_rms": float(reason_direct_rms), "reason_reread_delta_rms": float(output["reason_reread_delta_effective"].pow(2).mean().sqrt()), "reason_reread_delta_raw_rms": float(output["reason_reread_delta_raw"].pow(2).mean().sqrt()), "reason_exchange_delta_rms": float(output["reason_exchange_delta_effective"].pow(2).mean().sqrt()), "reason_exchange_delta_raw_rms": float(output["reason_exchange_delta_raw"].pow(2).mean().sqrt()), "reason_latent_delta_raw_rms": float(output["reason_latent_delta_raw"].pow(2).mean().sqrt()), "reason_latent_delta_effective_rms": float(output["reason_latent_delta_effective"].pow(2).mean().sqrt()), "reason_reread_to_direct_ratio": float(output["reason_reread_delta_effective"].pow(2).mean().sqrt().div(reason_direct_rms).item()), "reason_exchange_to_direct_ratio": float(output["reason_exchange_delta_effective"].pow(2).mean().sqrt().div(reason_direct_rms).item()), "annotation_delta_rms": float(output["annotation_delta_effective"].pow(2).mean().sqrt()), "annotation_delta_raw_rms": float(output["annotation_delta_raw"].pow(2).mean().sqrt()), "threshold_logit_raw_rms": float(output["threshold_logit_raw"].pow(2).mean().sqrt()), "threshold_logit_effective_rms": float(output["threshold_logit_effective"].pow(2).mean().sqrt()), "curriculum_state": model.curriculum_state(), "owner_active": owner_active, "owner_step_counts": dict(optimizer_step_counts), "owner_lrs": {owner: float(item.param_groups[0]["lr"]) for owner, item in optimizers.items()}, "curriculum_sha256": curriculum_digest, "explicit_reliability_mean": float(output["evidence_reliability"].mean().item()), "explicit_reliability_std": float(output["evidence_reliability"].std().item()), "explicit_strong_rate": float((output["evidence_reliability"] >= 0.7).float().mean()), "explicit_unreliable_rate": float((output["evidence_reliability"] < 0.3).float().mean()), "latent_token_norm": float(output["latent_evidence_tokens"].norm(dim=-1).mean()), "view_consistency_mean": float(model.evidence_fields.view_consistency_ema.mean()), "overlap_mean": float(output["exchange_overlap"].mean()), "overlap_max": float(output["exchange_overlap"].max()), "gate_mean": float(output["exchange_gate"].mean()), "gate_max": float(output["exchange_gate"].max()), "gate_active_rate_gt_0p1": float((output["exchange_gate"] > 0.1).float().mean()), "wrong_target_message_ratio": float(output["wrong_target_message_ratio"]), "reference_center_collapse_rate": float(output["center_collapse_rate"].item()), "reference_out_of_bounds_rate": float(output["out_of_bounds_rate"].item()), "hard_sample_improvement_rate": float(hard_improvement), "easy_sample_regression_rate": float(easy_regression), "grad_action_foundation": last_owner_stats.get("action_foundation", {}).get("grad_norm_pre_clip", 0.0), "grad_action_decoder": last_owner_stats.get("action_decoder", {}).get("grad_norm_pre_clip", 0.0), "grad_reason": last_owner_stats.get("reason_semantic", {}).get("grad_norm_pre_clip", 0.0), "grad_evidence_grounding": float(tensor_list_norm(grounding_credit)), "grad_evidence_target_credit_raw": float(tensor_list_norm(raw_credit)), "grad_evidence_target_credit_projected": float(tensor_list_norm(projected_credit)), "grad_annotation": last_owner_stats.get("annotation_adapter", {}).get("grad_norm_pre_clip", 0.0), "owner_diagnostics": last_owner_stats, **firewall_report, **gpu_memory_gb(device), **ownership_snapshot(model)})
                record.update({
                    "dino_call_count_batch": dino_call_count_batch,
                    "pcvl_optimizer_step_count": pcvl_optimizer_step_count,
                    "pcvl_nonzero_update_count": pcvl_nonzero_update_count,
                    "pcvl_grad_norm": last_pcvl_update["grad_norm"],
                    "pcvl_parameter_delta_norm": last_pcvl_update["parameter_delta_norm"],
                    "gpu_peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0,
                })
                family_indices: dict[str, list[int]] = {}
                for field_index, field in enumerate(active_fields):
                    family_indices.setdefault(str(field["family"]), []).append(field_index)
                presence_probs = torch.sigmoid(output["evidence_presence_logits"])
                observability_probs = torch.sigmoid(output["evidence_observability_logits"])
                reference_variance = output["reference_point_variance"].detach()
                record.update({
                    "per_family_presence_rate": {family: float(presence_probs[:, indices].mean()) for family, indices in family_indices.items()},
                    "per_family_observability_rate": {family: float(observability_probs[:, indices].mean()) for family, indices in family_indices.items()},
                    "prototype_margin_mean": float(output["evidence_prototype_margin"].mean()),
                    "evidence_attention_entropy": float(output["evidence_attention_entropy"]),
                    "evidence_effective_support": float(output["evidence_effective_support"]),
                    "action_reason_message_norm": float(output["action_reason_message_norm"]),
                    "reason_action_message_norm": float(output["reason_action_message_norm"]),
                    "reason_token_shuffle_delta": float(output["reason_token_shuffle_delta"]),
                    "evidence_shuffle_delta": float(output["evidence_shuffle_delta"]),
                    "reference_point_variance_x": float(reference_variance[0]),
                    "reference_point_variance_y": float(reference_variance[1]),
                })
                append_jsonl(output_dir / "loss_components.jsonl", record)
                append_jsonl(output_dir / "gradient_ownership.jsonl", {"epoch": epoch, "optimizer_step": global_optimizer_step, "owners": last_owner_stats, **firewall_report})
                append_jsonl(output_dir / "mechanism_batch_stats.jsonl", {key: record[key] for key in ("epoch", "optimizer_step", "dino_call_count_batch", "pcvl_optimizer_step_count", "pcvl_nonzero_update_count", "pcvl_grad_norm", "pcvl_parameter_delta_norm", "gpu_peak_reserved_gb", "action_exchange_to_direct_ratio", "reason_exchange_to_direct_ratio", "action_reread_to_direct_ratio", "reason_reread_to_direct_ratio", "explicit_reliability_mean", "explicit_reliability_std", "gate_active_rate_gt_0p1", "reference_center_collapse_rate", "selected_effect_mean", "control_effect_mean", "wrong_effect_mean", "intervention_pair_count", "annotation_delta_rms")})
                print(json.dumps(record, sort_keys=True), flush=True)
            last_output = output
            last_batch_finished = time.perf_counter()
        # Never drop an incomplete accumulated tail batch.
        if len(train_loader) % args.gradient_accumulation_steps:
            last_owner_stats = _step_owners(model, representation_owners, optimizers, schedulers, config, optimizer_step_counts, owner_active)
            global_optimizer_step += 1
        # CalAlign is trained only on the deterministic train-calib split.
        threshold_values = []
        threshold_teacher_before = model.threshold_head.theta_teacher.detach().clone()
        model.threshold_head.train()
        for calib_batch in calib_loader if owner_active["threshold_head"] else ():
            with torch.no_grad():
                raw = model(calib_batch["image"].to(device, non_blocking=True))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                threshold = model._threshold_with_activation(raw["action_logits_final_raw"].detach(), raw["reason_logits_observed"].detach())
                threshold_loss = F.binary_cross_entropy_with_logits(threshold["action_logits_deploy"], calib_batch["action"].to(device)) + F.binary_cross_entropy_with_logits(threshold["reason_logits_deploy"], calib_batch["reason"].to(device))
            optimizers["threshold_head"].zero_grad(set_to_none=True)
            threshold_loss.backward()
            clip_grad_norm_(parameter_ownership(model)["threshold_head"], float(config["training"]["grad_clip_reason"]))
            optimizers["threshold_head"].step()
            schedulers["threshold_head"].step()
            optimizer_step_counts["threshold_head"] += 1
            threshold_values.append(float(threshold_loss.detach()))
        if threshold_values:
            last_threshold_loss = sum(threshold_values) / len(threshold_values)
            if float(model.curriculum_state()["threshold"]) >= 1.0:
                model.threshold_head.update_teacher(model.threshold_head.compose_theta().detach(), ema=1.0)
        epoch_step_deltas = {owner: optimizer_step_counts[owner] - epoch_start_step_counts[owner] for owner in optimizers}
        protocol_errors = []
        for owner, active in owner_active.items():
            if active and epoch_step_deltas[owner] <= 0:
                protocol_errors.append(f"active owner {owner} did not step")
            if not active and epoch_step_deltas[owner] != 0:
                protocol_errors.append(f"inactive owner {owner} advanced by {epoch_step_deltas[owner]}")
            if not active and optimizers[owner].state:
                protocol_errors.append(f"inactive owner {owner} has optimizer state")
            if int(schedulers[owner].last_epoch) != optimizer_step_counts[owner]:
                protocol_errors.append(
                    f"owner {owner} scheduler clock {schedulers[owner].last_epoch} "
                    f"does not match steps {optimizer_step_counts[owner]}"
                )
        if float(model.curriculum_state()["threshold"]) < 1.0 and not torch.equal(
            threshold_teacher_before, model.threshold_head.theta_teacher
        ):
            protocol_errors.append("threshold teacher mutated before full threshold activation")
        if epoch >= 6 and not all(owner_active.values()):
            protocol_errors.append("safe_joint epoch does not activate every owner")
        if protocol_errors:
            write_json(output_dir / f"curriculum_protocol_error_epoch_{epoch:03d}.json", {"epoch": epoch, "curriculum_state": model.curriculum_state(), "owner_step_deltas": epoch_step_deltas, "errors": protocol_errors})
            raise RuntimeError("; ".join(protocol_errors))
        metrics, tensors = evaluate_precise(model, test_loader, device)
        if pcvl_probes is not None and audit_loader is not None:
            def audit_target_provider(audit_batch, audit_device):
                return grounding_adapter.stack_batch([train_grounding[name] for name in audit_batch["file_name"]], audit_device)
            pcvl_identity_keys = (
                "git_head", "source_tree_sha256", "config_sha256", "skill_sha256",
                "pretrained_weights_sha256", "action_schema_sha256",
                "train_audit_indices_sha256", "train_audit_file_names_sha256",
            )
            evaluate_pcvl(
                pcvl_probes, model, audit_loader, audit_target_provider, device, output_dir / "pcvl",
                provenance={**{key: run_manifest[key] for key in pcvl_identity_keys}, "epoch": epoch},
            )
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        write_json(epoch_dir / "metrics_summary.json", metrics)
        write_json(epoch_dir / "branch_metrics.json", metrics)
        write_json(epoch_dir / "gradient_firewall.json", firewall_report)
        write_json(epoch_dir / "counterfactual_stats.json", metrics["counterfactual"])
        write_json(epoch_dir / "per_action_metrics.json", metrics["action_deploy"])
        write_json(epoch_dir / "per_reason_metrics.json", {"observed": metrics["reason_observed"], "semantic": metrics["reason_semantic"], "deploy": metrics["reason_deploy"]})
        mechanism_test = metrics["mechanism_test"]
        write_json(epoch_dir / "evidence_family_stats.json", {"aggregation": "full_test", "presence_mean": mechanism_test["presence_mean"], "observability_mean": mechanism_test["observability_mean"], "actor_part_occupancy_mean": mechanism_test["actor_part_occupancy_mean"], "actor_part_type_mean": mechanism_test["actor_part_type_mean"]})
        write_json(epoch_dir / "evidence_reliability.json", {"aggregation": "full_test", "mean": mechanism_test["reliability_mean"], "strong_rate": mechanism_test["reliability_strong_rate"], "weak_rate": mechanism_test["reliability_weak_rate"]})
        write_json(epoch_dir / "exchange_stats.json", {"aggregation": "full_test", "overlap_mean": mechanism_test["overlap_mean"], "gate_mean": mechanism_test["gate_mean"], "gate_active_rate_gt_0p1": mechanism_test["gate_active_rate_gt_0p1"]})
        write_json(epoch_dir / "reread_stats.json", {"aggregation": "full_test", "center_collapse_rate": mechanism_test["center_collapse_rate"], "out_of_bounds_rate": mechanism_test["out_of_bounds_rate"], "reference_variance": mechanism_test["reference_variance"]})
        write_json(epoch_dir / "annotation_gap.json", {"aggregation": "full_test", "annotation_delta_rms": mechanism_test["annotation_delta_rms"], "semantic_observed_gap_rms": mechanism_test["semantic_observed_gap_rms"]})
        save_epoch_tensors(epoch_dir, {name: value for name, value in tensors.items() if isinstance(value, torch.Tensor) and (name.startswith("action_") or name.startswith("reason_"))}, tensors["labels_action"], tensors["labels_reason"])
        write_json(epoch_dir / "file_names.json", {"file_names": tensors["file_names"]})
        action_errors = F.binary_cross_entropy_with_logits(tensors["action_deploy"], tensors["labels_action"], reduction="none").sum(-1)
        reason_errors = F.binary_cross_entropy_with_logits(tensors["reason_deploy"], tensors["labels_reason"], reduction="none").sum(-1)
        worst_indices = (action_errors + reason_errors).topk(min(32, len(action_errors))).indices.tolist()
        for index in worst_indices:
            name = tensors["file_names"][index]
            action_error = F.binary_cross_entropy_with_logits(tensors["action_deploy"][index], tensors["labels_action"][index], reduction="sum")
            reason_error = F.binary_cross_entropy_with_logits(tensors["reason_deploy"][index], tensors["labels_reason"][index], reduction="sum")
            append_jsonl(epoch_dir / "failure_cases.jsonl", {"file_name": name, "action_bce": float(action_error), "reason_bce": float(reason_error)})
        for index, name in enumerate(tensors["file_names"][:32]):
            append_jsonl(epoch_dir / "evidence_cases.jsonl", {"file_name": name, "reliability": tensors["evidence_reliability"][index].tolist(), "presence": tensors["evidence_presence"][index].tolist(), "part_coordinates": tensors["evidence_coordinates"][index].tolist()})
        export_precise_cases(epoch_dir / "case_exports", tensors["case_rows"])
        curriculum_epoch_record = {
            "epoch": epoch,
            "curriculum_state": model.curriculum_state(),
            "curriculum_sha256": curriculum_digest,
            "owner_active": owner_active,
            "owner_step_deltas": epoch_step_deltas,
            "owner_step_counts": dict(optimizer_step_counts),
            "owner_scheduler_last_epoch": {
                owner: int(item.last_epoch) for owner, item in schedulers.items()
            },
            "owner_optimizer_state_nonempty": {
                owner: bool(item.state) for owner, item in optimizers.items()
            },
            "owner_lrs": {owner: float(item.param_groups[0]["lr"]) for owner, item in optimizers.items()},
            "threshold_teacher_updated": bool(
                not torch.equal(threshold_teacher_before, model.threshold_head.theta_teacher)
            ),
        }
        write_json(epoch_dir / "curriculum_state.json", curriculum_epoch_record)
        append_jsonl(output_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics, **curriculum_epoch_record, "pcvl_optimizer_step_count": pcvl_optimizer_step_count, "pcvl_nonzero_update_count": pcvl_nonzero_update_count})
        score_values = {"action_mf1": metrics["action_deploy"]["Act_mF1"], "exp_mf1": metrics["reason_deploy"]["Exp_mF1"], "exp_map": metrics["reason_deploy"]["Exp_mAP"], "semantic_exp_map": metrics["reason_semantic"]["Exp_mAP"]}
        improved = {name: value > best_scores[name] for name, value in score_values.items()}
        for name, value in score_values.items():
            if improved[name]:
                best_scores[name] = value
        joint_improved = metrics["deploy_fixed_joint"] > best
        if joint_improved:
            best = metrics["deploy_fixed_joint"]
        checkpoint = {"model": model.state_dict(), "epoch": epoch, "optimizers": {owner: item.state_dict() for owner, item in optimizers.items()}, "schedulers": {owner: item.state_dict() for owner, item in schedulers.items()}, "rng_state": torch.get_rng_state(), "python_rng_state": random.getstate(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "global_optimizer_step": global_optimizer_step, "global_micro_step": global_micro_step, "optimizer_step_counts": optimizer_step_counts, "best_deploy_joint": best, "best_scores": best_scores, "threshold_teacher": model.threshold_head.theta_teacher.detach().cpu(), "view_consistency_ema": model.evidence_fields.view_consistency_ema.detach().cpu(), "active_field_schema": [field["name"] for field in active_fields], "implementation_fingerprint": fingerprint, "curriculum_state": model.curriculum_state(), "curriculum_sha256": curriculum_digest, "owner_active_epochs": active_epochs_by_owner, "owner_step_deltas": epoch_step_deltas}
        checkpoint["pcvl_optimizer_step_count"] = pcvl_optimizer_step_count
        checkpoint["pcvl_nonzero_update_count"] = pcvl_nonzero_update_count
        if pcvl_probes is not None and pcvl_optimizer is not None:
            checkpoint["pcvl_probes"] = pcvl_probes.state_dict()
            checkpoint["pcvl_optimizer"] = pcvl_optimizer.state_dict()
        torch.save(checkpoint, output_dir / "checkpoint_latest.pth")
        if joint_improved:
            torch.save(checkpoint, output_dir / "checkpoint_best_test_deploy_joint.pth")
        checkpoint_names = {"action_mf1": "checkpoint_best_test_action_mf1.pth", "exp_mf1": "checkpoint_best_test_exp_mf1.pth", "exp_map": "checkpoint_best_test_exp_map.pth", "semantic_exp_map": "checkpoint_best_test_semantic_exp_map.pth"}
        for name, was_improved in improved.items():
            if was_improved:
                torch.save(checkpoint, output_dir / checkpoint_names[name])


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
    parser.add_argument("--mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--allow_full_with_embedded_curriculum", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
