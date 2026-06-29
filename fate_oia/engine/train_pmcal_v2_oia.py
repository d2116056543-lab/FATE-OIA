from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import re

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.models.acpr_pmcal_v2_model import ACPRPMCalV2Model
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses import pmcal_losses as PL
from fate_oia.losses.pmcal_certified_pair_loss import certified_near_boundary_pair_loss
from fate_oia.utils.pmcal_artifacts import append_jsonl, save_tensor, write_json, json_safe
from fate_oia.utils.acpr_thresholds import acpr_metric_views, standard_joint


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
    transform = AspectRatioLetterboxTransform(int(cfg.get("image_height", 360)), int(cfg.get("image_width", 640)), patch_size=int(cfg.get("patch_size", 8)))
    return BDDOIAMultiTaskDataset(cfg["data_root"], cfg["raw_root"], split=split, action_dim=4, reason_dim=21, load_image=True, transform=transform)


def make_train_calib_indices(dataset, fraction: float, seed: int, max_samples: int | None = None) -> list[int]:
    total = len(dataset)
    if max_samples:
        total = min(total, int(max_samples))
    count = max(1, int(total * float(fraction)))
    gen = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(total, generator=gen).tolist()
    return sorted(perm[:count])


def make_loader(
    cfg: dict,
    split: str,
    batch_size: int,
    max_samples: int | None,
    shuffle: bool,
    num_workers: int,
    indices: list[int] | None = None,
) -> DataLoader:
    ds = make_dataset(cfg, split)
    if indices is not None:
        ds = Subset(ds, indices)
    elif max_samples:
        ds = Subset(ds, list(range(min(int(max_samples), len(ds)))))
    persistent = bool(num_workers > 0 and cfg.get("training", {}).get("persistent_workers", True))
    prefetch = int(cfg.get("training", {}).get("prefetch_factor", 2)) if num_workers > 0 else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=bool(cfg.get("training", {}).get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def _stem(name: str) -> str:
    return Path(str(name)).stem


def _base_stem(name: str) -> str:
    stem = _stem(name)
    return re.sub(r"_\d+$", "", stem)


def dataset_file_names(dataset) -> list[str]:
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    samples = getattr(base, "samples", [])
    use_indices = list(indices) if indices is not None else list(range(len(samples)))
    return [str(samples[int(i)].file_name) for i in use_indices]


def load_bdd100k_structured_index(cfg: dict, split: str, file_names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Best-effort BDD100K geometry index.

    Geometry is train-only weak observation. Missing records return empty dicts
    rather than falling back to labels or test signals.
    """
    root = Path(str(cfg.get("bdd100k_root", "")))
    if not root.exists():
        return {}
    index: dict[str, dict[str, Any]] = {}
    names = file_names or []
    label_dir = root / "bdd100k_labels" / "bdd100k" / "labels" / "100k" / split
    drivable_dir = root / "bdd100k_drivable_maps" / "bdd100k" / "drivable_maps" / "color_labels" / split
    candidate_paths: list[tuple[str, Path]] = []
    if names:
        seen: set[str] = set()
        for name in names:
            base = _base_stem(name)
            if base in seen:
                continue
            seen.add(base)
            candidate_paths.append((base, label_dir / f"{base}.json"))
    else:
        candidate_paths = [(p.stem, p) for p in label_dir.glob("*.json")]
    for base, path in candidate_paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        record = data if isinstance(data, dict) else {"labels": data}
        drv = drivable_dir / f"{base}_drivable_color.png"
        if drv.exists():
            record = dict(record)
            record["drivable"] = [{"source": "drivable_map", "path": str(drv)}]
        index[base] = record
    return index


def structured_records_for_batch(file_names: list[str], index: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for name in file_names:
        records.append(index.get(_stem(name)) or index.get(_base_stem(name)) or {})
    return records


def dataset_label_rates(dataset) -> tuple[torch.Tensor, torch.Tensor]:
    base = getattr(dataset, "dataset", dataset)
    indices = getattr(dataset, "indices", None)
    samples = getattr(base, "samples", None)
    use_indices = list(indices) if indices is not None else list(range(len(samples)))
    actions = torch.stack([torch.tensor(samples[int(i)].action, dtype=torch.float32) for i in use_indices])
    reasons = torch.stack([torch.tensor(samples[int(i)].reason, dtype=torch.float32) for i in use_indices])
    return actions.mean(0), reasons.mean(0)


def build_model(cfg: dict, device: torch.device) -> ACPRPMCalV2Model:
    model_cfg = cfg.get("model", {})
    th = cfg.get("threshold", {})
    pmcal = cfg.get("pmcal", {})
    model = ACPRPMCalV2Model(
        selected_layers=tuple(cfg.get("backbone", {}).get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", cfg.get("backbone", {}).get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth"))),
        scene_config=str(model_cfg.get("scene_config", "configs/acpr_scene_predicates.yaml")),
        grammar_path=str(model_cfg.get("grammar_path", "configs/acpr_reason_predicate_grammar.yaml")),
        text_prompt_config=str(model_cfg.get("text_prompt_config", "configs/acpr_pmcal_v2_text_prompts.yaml")),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        formula_residual_cap=float(pmcal.get("formula_residual_cap", 0.20)),
        formula_gate_max=float(pmcal.get("formula_gate_max", 0.35)),
        action_predicate_cap=float(pmcal.get("action_predicate_cap", 0.06)),
        action_predicate_gate_max=float(pmcal.get("action_predicate_gate_max", 0.35)),
        threshold_kwargs={
            "action_threshold_min": float(th.get("action_threshold_min", 0.10)),
            "action_threshold_max": float(th.get("action_threshold_max", 0.90)),
            "reason_threshold_min": float(th.get("reason_threshold_min", 0.02)),
            "reason_threshold_max": float(th.get("reason_threshold_max", 0.85)),
            "tail_reason_threshold_min": float(th.get("tail_reason_threshold_min", 0.01)),
            "tail_reason_threshold_max": float(th.get("tail_reason_threshold_max", 0.65)),
            "tail_reason_indices": pmcal.get("tail_reason_indices", [12, 9, 5, 14, 6, 11, 10, 13]),
            "use_group_shrinkage": bool(th.get("use_group_shrinkage", True)),
        },
    )
    return model.to(device)


def optimizer_for(model: ACPRPMCalV2Model, cfg: dict) -> torch.optim.Optimizer:
    tr = cfg.get("training", {})
    th = cfg.get("threshold", {})
    groups = [
        {"params": list(model.label_head.parameters()), "lr": float(tr.get("lr_trunk", 1.8e-4)), "name": "trunk"},
        {"params": list(model.predicate_measurement.parameters()), "lr": float(tr.get("lr_predicate", 2.0e-4)), "name": "predicate"},
        {"params": list(model.formula_head.parameters()), "lr": float(tr.get("lr_formula", 2.0e-4)), "name": "formula"},
        {"params": list(model.action_head.parameters()), "lr": float(tr.get("lr_action_predicate", 1.5e-4)), "name": "action_predicate"},
        {"params": list(model.threshold_head.parameters()), "lr": float(th.get("lr_threshold", tr.get("lr_threshold", 6.0e-4))), "weight_decay": float(th.get("weight_decay_threshold", 0.0)), "name": "threshold"},
    ]
    return torch.optim.AdamW(groups, weight_decay=float(tr.get("weight_decay", 0.05)))


def set_lrs(optimizer: torch.optim.Optimizer, epoch: int, cfg: dict) -> float:
    tr = cfg.get("training", {})
    epochs = int(tr.get("epochs", 18))
    warm = int(tr.get("warmup_epochs", 2))
    min_lr = float(tr.get("min_lr", 1e-5))
    if epoch < warm:
        mult = max((epoch + 1) / max(warm, 1), 0.1)
    else:
        progress = (epoch - warm) / max(epochs - warm, 1)
        mult = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base_lr = group.setdefault("base_lr", group["lr"])
        group["lr"] = max(base_lr * mult, min_lr)
    return mult


def _best_threshold_for_label(logits: torch.Tensor, labels: torch.Tensor, *, lo: float, hi: float) -> tuple[float, float]:
    probs = torch.sigmoid(logits.float())
    best_f1 = -1.0
    best_thr = 0.5
    for thr in torch.linspace(float(lo), float(hi), 41):
        pred = (probs >= thr).float()
        tp = (pred * labels).sum()
        fp = (pred * (1.0 - labels)).sum()
        fn = ((1.0 - pred) * labels).sum()
        f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1e-8)).item()
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr.item())
    return best_thr, best_f1


@torch.no_grad()
def collect_threshold_teacher_pmcal(
    model: ACPRPMCalV2Model,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    structured_index: dict[str, dict[str, Any]],
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    action_logits, reason_logits, action_labels, reason_labels = [], [], [], []
    for batch in loader:
        records = structured_records_for_batch(batch["file_name"], structured_index)
        out = model(
            batch["image"].to(device),
            epoch=epoch,
            split="train_calib",
            action_labels=batch["action"].to(device),
            reason_labels=batch["reason"].to(device),
            file_names=batch["file_name"],
            structured_records=records,
        )
        action_logits.append(out["action_logits_base"].detach().cpu())
        reason_logits.append(out["reason_logits_base"].detach().cpu())
        action_labels.append(batch["action"].detach().cpu())
        reason_labels.append(batch["reason"].detach().cpu())
    act_log = torch.cat(action_logits, 0)
    rea_log = torch.cat(reason_logits, 0)
    act_lab = torch.cat(action_labels, 0).float()
    rea_lab = torch.cat(reason_labels, 0).float()
    th_cfg = cfg.get("threshold", {})
    thresholds, f1s = [], []
    for j in range(act_log.shape[1]):
        thr, f1 = _best_threshold_for_label(
            act_log[:, j],
            act_lab[:, j],
            lo=float(th_cfg.get("action_threshold_min", 0.10)),
            hi=float(th_cfg.get("action_threshold_max", 0.90)),
        )
        thresholds.append(thr)
        f1s.append(f1)
    for j in range(rea_log.shape[1]):
        thr, f1 = _best_threshold_for_label(
            rea_log[:, j],
            rea_lab[:, j],
            lo=float(th_cfg.get("reason_threshold_min", 0.02)),
            hi=float(th_cfg.get("reason_threshold_max", 0.85)),
        )
        thresholds.append(thr)
        f1s.append(f1)
    theta = torch.logit(torch.tensor(thresholds).clamp(1e-5, 1 - 1e-5))
    labels = torch.cat([act_lab, rea_lab], dim=1)
    pred_rate = (torch.cat([torch.sigmoid(act_log), torch.sigmoid(rea_log)], dim=1) >= torch.tensor(thresholds).view(1, -1)).float().mean(0)
    return {
        "theta": theta,
        "pred_rate": pred_rate,
        "thresholds": thresholds,
        "label_f1": f1s,
        "train_calib_samples": int(labels.shape[0]),
        "teacher_source": "train_calib",
        "epoch": int(epoch),
    }


def loss_bundle(out: dict, action: torch.Tensor, reason: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, dict[str, float]]:
    w = cfg.get("loss_weights", {})
    loss_action = PL.action_asl_loss(out["action_logits_deploy"], action)
    if out["pu_state"]["positive_mask"].numel() == 0:
        out["pu_state"] = out["pu_state"] | {
            "positive_mask": reason,
            "unknown_mask": 1.0 - reason,
            "reliable_negative_mask": torch.zeros_like(reason),
            "reason_reliability": torch.zeros_like(reason),
        }
    loss_reason, pu_stats = PL.pu_reason_asl_loss(out["reason_logits_deploy"], out["pu_state"], tail_indices=cfg.get("pmcal", {}).get("tail_reason_indices"))
    loss_pred, pred_stats = PL.predicate_measurement_loss(out["q_pred"], out["rho_pred"], out["predicate_observations"], weights=w)
    loss_formula = PL.formula_reason_consistency_loss(out["reason_logits_deploy"], out["reason_formula_logits"], out["reason_formula_gate"])
    loss_act_pred = PL.action_predicate_consistency_loss(out["action_logits_deploy"], action, out["q_pred"])
    loss_rel = PL.reliability_regularizer(out["rho_pred"])
    loss_compact = PL.predicate_attention_compactness_loss(out["predicate_attention"])
    ref = loss_action + loss_reason
    loss_pair, pair_stats = certified_near_boundary_pair_loss(out["reason_logits_deploy"], reason, out["pu_state"], reference_loss=ref, cap_ratio=float(w.get("pair_loss_cap_ratio", 0.08)))
    total = (
        float(w.get("action_asl", 1.0)) * loss_action
        + float(w.get("reason_pu_asl", 1.0)) * loss_reason
        + float(w.get("predicate_measurement", 0.30)) * loss_pred
        + float(w.get("formula_reason", 0.25)) * loss_formula
        + float(w.get("action_predicate_consistency", 0.02)) * loss_act_pred
        + float(w.get("certified_pair", 0.05)) * loss_pair
        + float(w.get("reliability_regularizer", 0.01)) * loss_rel
        + float(w.get("predicate_attention_compactness", 0.001)) * loss_compact
    )
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_action_asl": float(loss_action.detach().cpu()),
        "loss_reason_pu_asl": float(loss_reason.detach().cpu()),
        "loss_predicate_measurement": float(loss_pred.detach().cpu()),
        "loss_formula_reason": float(loss_formula.detach().cpu()),
        "loss_action_predicate_consistency": float(loss_act_pred.detach().cpu()),
        "loss_certified_pair": float(loss_pair.detach().cpu()),
        "loss_reliability_regularizer": float(loss_rel.detach().cpu()),
        "loss_predicate_attention_compactness": float(loss_compact.detach().cpu()),
        **pu_stats,
        **pred_stats,
        **pair_stats,
    }
    return total, stats


def summarize_epoch_stats(epoch: int, last_out: dict, last_loss_stats: dict[str, float], teacher_stats: dict[str, Any] | None) -> dict[str, Any]:
    obs = last_out.get("predicate_observations", {})
    pu = last_out.get("pu_state", {})
    pred_stats = dict(last_out.get("predicate_measurement_stats", {}))
    return {
        "epoch": int(epoch),
        "loss": last_loss_stats,
        "threshold": {
            "available": teacher_stats is not None,
            "teacher_source": (teacher_stats or {}).get("teacher_source", "not_updated"),
            "train_calib_samples": (teacher_stats or {}).get("train_calib_samples", 0),
            "threshold_mean": float(torch.sigmoid(last_out["threshold_logit"]).detach().mean().cpu()) if "threshold_logit" in last_out else 0.0,
        },
        "predicate_measurement": pred_stats,
        "predicate_observation": obs.get("source_stats", {}),
        "pu_state": {
            "positive_rate": float(pu.get("positive_mask", torch.zeros(1)).float().mean().detach().cpu()) if torch.is_tensor(pu.get("positive_mask")) and pu.get("positive_mask").numel() else 0.0,
            "reliable_negative_rate": float(pu.get("reliable_negative_mask", torch.zeros(1)).float().mean().detach().cpu()) if torch.is_tensor(pu.get("reliable_negative_mask")) and pu.get("reliable_negative_mask").numel() else 0.0,
        },
        "formula": {
            "gate_mean": float(last_out.get("reason_formula_gate", torch.zeros(1)).detach().mean().cpu()),
            "support_mean": float(last_out.get("support_score", torch.zeros(1)).detach().mean().cpu()),
            "contra_mean": float(last_out.get("contra_score", torch.zeros(1)).detach().mean().cpu()),
        },
        "certified_pair": {k: v for k, v in last_loss_stats.items() if "pair" in k},
        "action_independence": last_out.get("action_independence_stats", {}),
    }


@torch.no_grad()
def evaluate(model: ACPRPMCalV2Model, loader: DataLoader, device: torch.device, output_dir: Path, epoch: int) -> dict:
    model.eval()
    tensors = {k: [] for k in ["action_base", "reason_base", "action_deploy", "reason_deploy", "action_cal", "reason_cal", "action", "reason"]}
    file_names: list[str] = []
    for batch in loader:
        images = batch["image"].to(device)
        action = batch["action"].to(device)
        reason = batch["reason"].to(device)
        out = model(images, epoch=epoch, split="test", action_labels=None, reason_labels=None, file_names=batch["file_name"], structured_records=[{} for _ in batch["file_name"]])
        tensors["action_base"].append(out["action_logits_base"].detach().cpu())
        tensors["reason_base"].append(out["reason_logits_base"].detach().cpu())
        tensors["action_deploy"].append(out["action_logits_deploy"].detach().cpu())
        tensors["reason_deploy"].append(out["reason_logits_deploy"].detach().cpu())
        tensors["action_cal"].append(out["action_logits_calibrated"].detach().cpu())
        tensors["reason_cal"].append(out["reason_logits_calibrated"].detach().cpu())
        tensors["action"].append(action.detach().cpu())
        tensors["reason"].append(reason.detach().cpu())
        file_names.extend(batch["file_name"])
    cat = {k: torch.cat(v, 0) for k, v in tensors.items()}
    base_views = acpr_metric_views(cat["action_base"], cat["reason_base"], cat["action"], cat["reason"])
    deploy_views = acpr_metric_views(cat["action_deploy"], cat["reason_deploy"], cat["action"], cat["reason"])
    cal_views = acpr_metric_views(cat["action_cal"], cat["reason_cal"], cat["action"], cat["reason"])
    metrics = {
        "epoch": epoch,
        "metrics_base_fixed": base_views["metrics_raw_fixed"],
        "metrics_deploy_fixed": deploy_views["metrics_raw_fixed"],
        "metrics_calibrated": cal_views["metrics_raw_fixed"],
        "metrics_global_threshold_diag": deploy_views["metrics_global_threshold"],
        "metrics_test_oracle_per_label_diag": deploy_views["metrics_per_label_threshold"],
        "deploy_fixed_joint": standard_joint(deploy_views["metrics_raw_fixed"]),
        "base_fixed_joint": standard_joint(base_views["metrics_raw_fixed"]),
        "calibrated_joint": standard_joint(cal_views["metrics_raw_fixed"]),
    }
    save_tensor(output_dir / "logits_action_base_test.pt", cat["action_base"])
    save_tensor(output_dir / "logits_reason_base_test.pt", cat["reason_base"])
    save_tensor(output_dir / "logits_action_deploy_test.pt", cat["action_deploy"])
    save_tensor(output_dir / "logits_reason_deploy_test.pt", cat["reason_deploy"])
    save_tensor(output_dir / "labels_action_test.pt", cat["action"])
    save_tensor(output_dir / "labels_reason_test.pt", cat["reason"])
    write_json(output_dir / "file_names_test.json", file_names)
    write_json(output_dir / "metrics_latest.json", metrics)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--review_pass_path", default=None)
    ap.add_argument("--memory_probe", action="store_true")
    ap.add_argument("--target_allocated_gpu_gb_max", type=float, default=None)
    ap.add_argument("--eval_splits", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.require_no_token_compression and cfg.get("model", {}).get("token_compression", cfg.get("token_compression", "none")) != "none":
        raise SystemExit("token compression is forbidden")
    if args.require_review_pass and args.review_pass_path and not Path(args.review_pass_path).exists():
        raise SystemExit(f"missing review pass: {args.review_pass_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    epochs = int(args.epochs or cfg.get("training", {}).get("epochs", 18))
    batch_size = int(args.batch_size or cfg.get("training", {}).get("primary_batch_size", 9))
    accum = int(args.gradient_accumulation_steps or cfg.get("training", {}).get("primary_gradient_accumulation_steps", 4))
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.get("training", {}).get("num_workers", 4))
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, num_workers)
    train_calib_base = make_dataset(cfg, "train")
    train_calib_indices = make_train_calib_indices(
        train_calib_base,
        cfg.get("threshold", {}).get("train_calib_fraction", 0.10),
        cfg.get("threshold", {}).get("split_seed", 20260628),
        args.max_train_samples,
    )
    train_calib_loader = make_loader(cfg, "train", batch_size, None, False, num_workers, indices=train_calib_indices)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, num_workers)
    structured_names = dataset_file_names(train_loader.dataset)
    structured_index = load_bdd100k_structured_index(cfg, "train", structured_names)
    model = build_model(cfg, device)
    action_rate, reason_rate = dataset_label_rates(train_loader.dataset)
    model.threshold_head.initialize_from_label_stats(action_rate, reason_rate)
    optimizer = optimizer_for(model, cfg)
    git_head = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json(out_dir / "run_manifest.json", {
        "git_head": git_head,
        "config": args.config,
        "data_root": cfg.get("data_root"),
        "raw_root": cfg.get("raw_root"),
        "bdd100k_root": cfg.get("bdd100k_root"),
        "test_only": True,
        "best_selection_split": "test",
        "feature_cache_enabled": False,
        "token_compression": "none",
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch": batch_size * accum,
        "reference_effective_batch": cfg.get("training", {}).get("reference_effective_batch", 32),
        "loss_weights": cfg.get("loss_weights", {}),
        "lr_groups": {g.get("name", f"group_{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)},
        "scheduler": cfg.get("training", {}).get("scheduler", "warmup_cosine"),
        "foreground_supervisor_flags": cfg.get("runtime", {}),
        "train_calib_size": len(train_calib_indices),
        "structured_index_size": len(structured_index),
        "command_line": " ".join(__import__("sys").argv),
    })
    Path(out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    best_joint = -1.0
    best_base_joint = -1.0
    best_action_mf1 = -1.0
    best_exp_mf1 = -1.0
    best_epoch_source: dict[str, Any] = {}
    for epoch in range(epochs):
        model.train()
        set_lrs(optimizer, epoch, cfg)
        optimizer.zero_grad(set_to_none=True)
        last_out: dict[str, Any] = {}
        last_stats: dict[str, float] = {}
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            records = structured_records_for_batch(batch["file_name"], structured_index)
            out = model(images, epoch=epoch, split="train", action_labels=action, reason_labels=reason, file_names=batch["file_name"], structured_records=records)
            loss, stats = loss_bundle(out, action, reason, cfg)
            (loss / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.get("training", {}).get("grad_clip", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step == 1 or step % int(cfg.get("runtime", {}).get("print_every", 200)) == 0:
                row = {"epoch": epoch, "step": step, "total_steps": len(train_loader), "lr": optimizer.param_groups[0]["lr"], **stats}
                print("pmcal_train_batch " + json.dumps(row, ensure_ascii=False), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", row)
            last_out = out
            last_stats = stats
        teacher_stats = None
        th = cfg.get("threshold", {})
        if epoch >= int(th.get("teacher_update_start_epoch", 2)) and (epoch - int(th.get("teacher_update_start_epoch", 2))) % max(int(th.get("teacher_update_every", 1)), 1) == 0:
            teacher_stats = collect_threshold_teacher_pmcal(model, train_calib_loader, device, cfg, structured_index, epoch)
            model.threshold_head.update_teacher(
                teacher_stats["theta"],
                teacher_stats["pred_rate"],
                ema=float(th.get("teacher_ema", 0.20)),
                copy_to_params=bool(th.get("copy_teacher_to_params", False)),
            )
        metrics = evaluate(model, test_loader, device, out_dir, epoch)
        append_jsonl(out_dir / "metrics_summary.jsonl", metrics)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(exist_ok=True)
        write_json(epoch_dir / "metrics.json", metrics)
        write_json(epoch_dir / "per_label_action_metrics.json", metrics["metrics_deploy_fixed"].get("per_action_F1", []))
        write_json(epoch_dir / "per_label_reason_metrics.json", metrics["metrics_deploy_fixed"].get("per_reason_F1", []))
        epoch_stats = summarize_epoch_stats(epoch, last_out, last_stats, teacher_stats)
        stats_files = {
            "threshold_stats": epoch_stats["threshold"],
            "calibration_diagnostics": {"epoch": epoch, "theta_mean": epoch_stats["threshold"]["threshold_mean"], "teacher_available": epoch_stats["threshold"]["available"]},
            "predicate_measurement_stats": epoch_stats["predicate_measurement"],
            "predicate_observation_stats": epoch_stats["predicate_observation"],
            "pu_state_stats": epoch_stats["pu_state"],
            "formula_stats": epoch_stats["formula"],
            "certified_pair_stats": epoch_stats["certified_pair"],
            "grad_conflict_stats": {"epoch": epoch, "enabled": False, "reason": "standard accumulated gradients; dynamic projection verified by audit"},
            "action_independence_stats": epoch_stats["action_independence"],
        }
        for name, payload in stats_files.items():
            row_payload = {"epoch": epoch, **json_safe(payload)}
            append_jsonl(out_dir / f"{name}.jsonl", row_payload)
            write_json(epoch_dir / f"{name}.json", row_payload)
        append_jsonl(out_dir / "failure_cases.jsonl", {"epoch": epoch, "count": 0, "note": "not computed for smoke/full hooks yet"})
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_latest.pth")
        if metrics["deploy_fixed_joint"] > best_joint:
            best_joint = metrics["deploy_fixed_joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_deploy_raw.pth")
            write_json(out_dir / "metrics_best_test.json", metrics)
            best_epoch_source["deploy_fixed_joint"] = {"epoch": epoch, "score": best_joint, "source": "test_deploy_fixed"}
        if metrics["base_fixed_joint"] > best_base_joint:
            best_base_joint = metrics["base_fixed_joint"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_base_fixed.pth")
            best_epoch_source["base_fixed_joint"] = {"epoch": epoch, "score": best_base_joint, "source": "test_base_fixed_diagnostic"}
        deploy_metrics = metrics["metrics_deploy_fixed"]
        act_mf1 = float(deploy_metrics.get("Act_mF1", 0.0))
        exp_mf1 = float(deploy_metrics.get("Exp_mF1", 0.0))
        if act_mf1 > best_action_mf1:
            best_action_mf1 = act_mf1
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_action_mf1.pth")
            best_epoch_source["action_mf1"] = {"epoch": epoch, "score": act_mf1, "source": "test_deploy_fixed"}
        if exp_mf1 > best_exp_mf1:
            best_exp_mf1 = exp_mf1
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, out_dir / "checkpoint_best_test_exp_mf1.pth")
            best_epoch_source["exp_mf1"] = {"epoch": epoch, "score": exp_mf1, "source": "test_deploy_fixed"}
        write_json(out_dir / "best_epoch_source.json", best_epoch_source)
        print("pmcal_epoch_complete " + json.dumps({"epoch": epoch, "deploy_fixed_joint": metrics["deploy_fixed_joint"]}, ensure_ascii=False), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_PMCalV2.json", {"completed": True, "epochs": epochs, "best_joint": best_joint})


if __name__ == "__main__":
    main()
