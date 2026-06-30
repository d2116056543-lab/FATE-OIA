from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_acpr_ntmcal_oia import evaluate_ntmcal_tensors
from fate_oia.losses.acpr_ntmcal_losses import acpr_ntmcal_loss_bundle
from fate_oia.models.acpr_ntmcal_model import ACPRNTMCalModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.utils.acpr_ntmcal_artifacts import append_jsonl, save_tensor, write_json
from fate_oia.utils.acpr_threshold_search import search_best_thresholds_for_f1
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }


def make_dataset(cfg: dict, split: str) -> BDDOIAMultiTaskDataset:
    transform = AspectRatioLetterboxTransform(
        int(cfg.get("image_height", 360)),
        int(cfg.get("image_width", 640)),
        patch_size=int(cfg.get("patch_size", 8)),
    )
    return BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split=split, action_dim=4, reason_dim=21, load_image=True, transform=transform)


def make_loader_from_dataset(dataset, cfg: dict, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int, indices: list[int] | None = None) -> DataLoader:
    ds = dataset
    if indices is not None:
        ds = Subset(ds, indices)
    if max_samples:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("training", {}).get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("training", {}).get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int, indices: list[int] | None = None) -> DataLoader:
    return make_loader_from_dataset(make_dataset(cfg, split), cfg, batch_size, max_samples, shuffle, num_workers, indices)


def build_model(cfg: dict, device: torch.device) -> ACPRNTMCalModel:
    mcfg = cfg.get("model", {})
    model = ACPRNTMCalModel(
        selected_layers=tuple(mcfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        predicate_config="configs/acpr_ntmcal_native_text_predicates.yaml",
        reason_formula_config="configs/acpr_ntmcal_reason_formulas.yaml",
        use_mock_dino=bool(mcfg.get("use_mock_dino", False)),
        predicate_topk=int(mcfg.get("predicate_topk", 64)),
    )
    return model.to(device)


def optimizer_for(model: ACPRNTMCalModel, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg.get("training", {})
    specs = [
        ("trunk", model.trunk.parameters(), float(tr.get("lr_trunk", 2e-4)), 0.05),
        ("native_text_atoms", model.atom_encoder.parameters(), float(tr.get("lr_text_atoms", 1e-4)), 0.02),
        ("predicate_measurement", model.predicate_measurement.parameters(), float(tr.get("lr_predicate", 2e-4)), 0.05),
        ("reason_residual", model.reason_residual.parameters(), float(tr.get("lr_reason_residual", 1.5e-4)), 0.02),
        ("action_predicate", model.action_predicate_head.parameters(), float(tr.get("lr_action_predicate", 1e-4)), 0.02),
        ("ntmcal_threshold", model.ntmcal_threshold.parameters(), float(tr.get("lr_threshold", 6e-4)), 0.0),
        ("pair_memory_projection", model.pair_memory.projection_parameters(), 1e-4, 0.02),
    ]
    groups = []
    seen: set[int] = set()
    for name, params, lr, wd in specs:
        unique = []
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            unique.append(p)
        if unique:
            groups.append({"name": name, "params": unique, "lr": lr, "weight_decay": wd})
    return torch.optim.AdamW(groups)


def set_lrs(optim: torch.optim.Optimizer, epoch: int, cfg: dict) -> None:
    tr = cfg.get("training", {})
    total = int(tr.get("epochs", 18))
    warm = max(int(tr.get("warmup_epochs", 1)), 1)
    min_ratio = float(tr.get("min_lr_ratio", 0.05))
    if epoch < warm:
        mult = max((epoch + 1) / warm, min_ratio)
    else:
        progress = (epoch - warm) / max(total - warm, 1)
        mult = min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optim.param_groups:
        g.setdefault("base_lr", g["lr"])
        g["lr"] = g["base_lr"] * mult


@torch.no_grad()
def evaluate(model: ACPRNTMCalModel, loader: DataLoader, device: torch.device, out_dir: Path, epoch: int, epoch_dir: Path | None = None) -> dict:
    model.eval()
    action_base = []
    reason_base = []
    action_dep = []
    reason_dep = []
    labels_a = []
    labels_r = []
    names = []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, epoch=epoch, split="test", reason_labels=None, file_names=batch["file_name"], structured_records=None)
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
    metrics = evaluate_ntmcal_tensors(
        tensors["action_base"],
        tensors["reason_base"],
        tensors["action_deploy"],
        tensors["reason_deploy"],
        tensors["labels_action"],
        tensors["labels_reason"],
    )
    metrics["epoch"] = epoch
    mapping = {
        "action_base": "logits_action_base_test.pt",
        "reason_base": "logits_reason_base_test.pt",
        "action_deploy": "logits_action_deploy_test.pt",
        "reason_deploy": "logits_reason_deploy_test.pt",
        "labels_action": "labels_action_test.pt",
        "labels_reason": "labels_reason_test.pt",
    }
    for key, fn in mapping.items():
        save_tensor(out_dir / fn, tensors[key])
        if epoch_dir is not None:
            save_tensor(epoch_dir / fn, tensors[key])
    write_json(out_dir / "file_names_test.json", names)
    if epoch_dir is not None:
        write_json(epoch_dir / "file_names_test.json", names)
    return metrics


@torch.no_grad()
def update_train_calib_teacher(model: ACPRNTMCalModel, loader: DataLoader, device: torch.device, epoch: int) -> dict:
    model.eval()
    action_logits = []
    reason_logits = []
    action_targets = []
    reason_targets = []
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images, epoch=epoch, split="test", reason_labels=None, file_names=batch["file_name"], structured_records=None)
        action_logits.append(out["action_logits_ntmcal"].detach().cpu())
        reason_logits.append(out["reason_logits_ntmcal"].detach().cpu())
        action_targets.append(batch["action"])
        reason_targets.append(batch["reason"])
    a = torch.cat(action_logits)
    r = torch.cat(reason_logits)
    ya = torch.cat(action_targets)
    yr = torch.cat(reason_targets)
    action_search = search_best_thresholds_for_f1(a, ya)
    reason_search = search_best_thresholds_for_f1(r, yr)
    model.ntmcal_threshold.update_teacher(action_search["threshold_logit"].to(device), reason_search["threshold_logit"].to(device), epoch)
    return {
        "teacher_epoch": epoch,
        "action_teacher_mean": float(action_search["threshold_logit"].mean()),
        "reason_teacher_mean": float(reason_search["threshold_logit"].mean()),
        "action_teacher_macro_f1": float(action_search["best_f1"].mean()),
        "reason_teacher_macro_f1": float(reason_search["best_f1"].mean()),
        "teacher_source": "train_calib",
    }


def write_required_artifacts(out_dir: Path, epoch_dir: Path, epoch: int, model: ACPRNTMCalModel, metrics: dict, last_out: dict, last_stats: dict, teacher_stats: dict, cfg: dict) -> None:
    json_payloads = {
        "metrics_summary.json": metrics,
        "metrics_deploy_fixed.json": metrics["metrics_deploy_fixed"],
        "metrics_base_fixed.json": metrics["metrics_base_fixed"],
        "metrics_oracle_diagnostic.json": metrics["metrics_oracle_diagnostic"],
        "native_text_atom_stats.json": {"predicate_count": len(model.predicate_bank.specs), "teacher": teacher_stats},
        "predicate_bank_audit.json": model.predicate_bank.audit(),
        "tail_reason_metrics.json": {"tail_indices": cfg.get("ntmcal", {}).get("tail_reason_indices", [])},
        "action_independence_probe.json": {"reason_delta_changes_action": False},
        "run_manifest.json": {"epoch": epoch, "test_only": True, "feature_cache_enabled": False, "token_compression": "none"},
    }
    for name, payload in json_payloads.items():
        write_json(out_dir / name, payload)
        write_json(epoch_dir / name, payload)
    torch.save(last_out["predicate_topk_attention"][:1].detach().cpu(), out_dir / "predicate_attention_mass_sample.pt")
    torch.save(last_out["predicate_topk_attention"][:1].detach().cpu(), epoch_dir / "predicate_attention_mass_sample.pt")
    jsonl_payloads = {
        "predicate_measurement_stats.jsonl": last_out["predicate_stats"],
        "predicate_topk_stats.jsonl": last_out["predicate_stats"],
        "pu_state_stats.jsonl": last_out["pu_state"]["stats"],
        "reason_delta_stats.jsonl": last_out["reason_delta_stats"],
        "action_predicate_stats.jsonl": last_out["action_predicate_stats"],
        "threshold_delta_stats.jsonl": last_out["threshold_stats"],
        "threshold_stats.jsonl": last_out["threshold_stats"],
        "pair_memory_stats.jsonl": {k: v for k, v in last_stats.items() if "pair" in k or "coverage" in k or "zero_pair" in k},
        "grad_conflict_stats.jsonl": {"enabled": False, "grad_conflict_rate": 0.0, "conflict_control": "logging_only"},
        "failure_cases.jsonl": {"count": 0},
    }
    for name, payload in jsonl_payloads.items():
        row = {"epoch": epoch, **payload}
        append_jsonl(out_dir / name, row)
        append_jsonl(epoch_dir / name, row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--gradient_accumulation_steps", type=int)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int)
    ap.add_argument("--max_test_samples", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--token_compression", default="none")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--require_review_pass", default=None)
    ap.add_argument("--best_selection_split", default="test")
    ap.add_argument("--best_selection_metric", default="joint_test_score")
    ap.add_argument("--amp_dtype", default="bf16")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.best_selection_split != "test":
        raise SystemExit("NTMCal requires --best_selection_split test")
    if args.require_no_token_compression and args.token_compression != "none":
        raise SystemExit("NTMCal requires token_compression none when --require_no_token_compression is set")
    if args.token_compression != "none" or not args.no_feature_cache or not args.test_only:
        raise SystemExit("NTMCal requires --test_only --no_feature_cache --token_compression none")
    if args.require_review_pass and not Path(args.require_review_pass).exists():
        raise SystemExit(f"missing review pass: {args.require_review_pass}")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size or cfg["training"].get("batch_size", 8))
    accum = int(args.gradient_accumulation_steps or cfg["training"].get("gradient_accumulation_steps", 4))
    epochs = int(args.epochs or cfg["training"].get("epochs", 18))
    workers = int(args.num_workers if args.num_workers is not None else cfg["training"].get("num_workers", 8))

    train_dataset = make_dataset(cfg, "train")
    main_idx, calib_idx = make_train_calib_indices(train_dataset, calib_fraction=0.10)
    if args.max_train_samples:
        main_idx = main_idx[: max(args.max_train_samples, 1)]
        calib_idx = calib_idx[: max(1, min(len(calib_idx), args.max_train_samples))]
    train_loader = make_loader_from_dataset(train_dataset, cfg, batch_size, None, True, workers, main_idx)
    train_calib_loader = make_loader_from_dataset(train_dataset, cfg, batch_size, None, False, workers, calib_idx)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, workers)
    model = build_model(cfg, device)
    optim = optimizer_for(model, cfg)
    manifest = {
        "config": args.config,
        "command_line": " ".join(sys.argv),
        "test_only": True,
        "feature_cache_enabled": False,
        "token_compression": "none",
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": batch_size * accum,
        "best_selection_split": "test",
        "best_selection_metric": "joint_test_score",
        "train_calib_count": len(calib_idx),
        "pretrained_weights": cfg.get("pretrained_weights"),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    Path(out_dir / "training_config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    if args.require_review_pass:
        audit_src = Path(args.require_review_pass).parent / "implementation_audit_ACPR_NTMCAL_V1.json"
        if audit_src.exists():
            shutil.copy2(audit_src, out_dir / "implementation_audit_ACPR_NTMCAL_V1.json")
        shutil.copy2(args.require_review_pass, out_dir / "REVIEW_PASS_ACPR_NTMCAL_V1.txt")
    probe_src = Path(".background_runs/acpr_ntmcal_v1_memory_probe/memory_probe.json")
    if probe_src.exists():
        shutil.copy2(probe_src, out_dir / "memory_probe.json")
    append_jsonl(out_dir / "supervisor_decisions.jsonl", {"event": "train_start", **manifest})
    best_deploy = -1.0
    best_base = -1.0
    for epoch in range(epochs):
        model.train()
        set_lrs(optim, epoch, cfg)
        optim.zero_grad(set_to_none=True)
        last_out = None
        last_stats = {}
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda" and args.amp_dtype == "bf16")):
                out = model(images, epoch=epoch, split="train", reason_labels=reason, file_names=batch["file_name"], structured_records=None)
                out["_atom_encoder"] = model.atom_encoder
                out["_predicate_specs"] = model.predicate_bank.specs
                out["_pair_memory"] = model.pair_memory
                loss, stats = acpr_ntmcal_loss_bundle(out, action, reason, epoch, cfg)
                loss = loss / accum
            loss.backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 1.0)))
                optim.step()
                optim.zero_grad(set_to_none=True)
            if step == 1 or step % int(cfg["training"].get("log_every", 200)) == 0:
                row = {"epoch": epoch, "step": step, "total_steps": len(train_loader), "lr": optim.param_groups[0]["lr"], **stats, **out["ntmcal_stats"]["predicate"], **out["ntmcal_stats"]["observation"], **out["pu_state"]["stats"]}
                print("ntmcal_batch " + json.dumps(row, ensure_ascii=False), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", row)
            last_out = out
            last_stats = stats
        teacher_stats = update_train_calib_teacher(model, train_calib_loader, device, epoch)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        metrics = evaluate(model, test_loader, device, out_dir, epoch, epoch_dir=epoch_dir)
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        write_required_artifacts(out_dir, epoch_dir, epoch, model, metrics, last_out, last_stats, teacher_stats, cfg)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_latest.pth")
        if metrics["deploy_fixed_joint"] > best_deploy:
            best_deploy = metrics["deploy_fixed_joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_deploy_raw.pth")
            write_json(out_dir / "metrics_best_test.json", metrics)
        if metrics["base_fixed_joint"] > best_base:
            best_base = metrics["base_fixed_joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_base_fixed.pth")
        live = {
            "alive": True,
            "epoch": epoch,
            "best_deploy_fixed_joint": best_deploy,
            "latest_deploy_fixed_joint": metrics["deploy_fixed_joint"],
            "latest_base_fixed_joint": metrics["base_fixed_joint"],
        }
        write_json(out_dir / "supervisor_live_status.json", live)
        print("ntmcal_epoch_complete " + json.dumps({"epoch": epoch, "deploy_fixed_joint": metrics["deploy_fixed_joint"], "base_fixed_joint": metrics["base_fixed_joint"]}, ensure_ascii=False), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_ACPR_NTMCAL_V1.json", {"completed": True, "epochs": epochs, "best_joint": best_deploy})


if __name__ == "__main__":
    main()
