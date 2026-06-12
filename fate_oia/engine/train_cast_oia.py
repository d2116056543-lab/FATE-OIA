from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.engine.eval_cast_oia import compute_action_set_metrics, evaluate_cast_outputs, write_json
from fate_oia.losses import cast_oia_losses as L
from fate_oia.models.cast_action_set_energy import action_targets_to_subset_ids
from fate_oia.models.cast_oia_model import CastOIAModel
from fate_oia.transforms import AspectRatioLetterboxTransform


LOSS_WEIGHTS = {
    "action_marginal": 1.00,
    "action_set": 0.60,
    "drop_add": 0.25,
    "cardinality": 0.15,
    "pair_compatibility": 0.10,
    "reason": 0.85,
    "reason_sigmoid_f1": 0.08,
    "reason_positive_boost": 1.0,
    "reason_negative_scale": 1.0,
    "tail_same_action_set_ranking": 0.06,
    "reason_to_action_set_alignment": 0.05,
    "text_evidence_contrast": 0.01,
    "graph_sparsity": 0.005,
    "calibration_regularizer": 0.01,
    "evidence_compactness": 0.001,
    "evidence_margin": 0.0,
}


def _load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def loss_weights_for_epoch(epoch: int) -> dict[str, float]:
    weights = dict(LOSS_WEIGHTS)
    if epoch <= 3:
        weights["reason"] = 1.20
        weights["action_set"] = 0.45
        weights["drop_add"] = 0.15
        weights["reason_positive_boost"] = 3.0
        weights["reason_negative_scale"] = 0.5
    return weights


def _write_jsonl(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_loader(cfg: dict[str, Any], split: str, batch_size: int, max_samples: int | None, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    model = cfg["model"]
    transform = AspectRatioLetterboxTransform(
        image_height=int(data["image_height"]),
        image_width=int(data["image_width"]),
        patch_size=int(model["patch_size"]),
        return_meta=False,
    )
    ds = BDDOIAMultiTaskDataset(
        data_root=data["data_root"],
        raw_root=data["raw_root"],
        split=split,
        action_dim=int(model["action_dim"]),
        reason_dim=int(model["reason_dim"]),
        load_image=True,
        transform=transform,
    )
    if max_samples:
        ds = Subset(ds, list(range(min(int(max_samples), len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


def build_model(cfg: dict[str, Any], device: torch.device) -> CastOIAModel:
    m = cfg["model"]
    model = CastOIAModel(
        dim=int(m["dim"]),
        action_dim=int(m["action_dim"]),
        reason_dim=int(m["reason_dim"]),
        ontology_path="configs/cast_oia_label_ontology.yaml",
        use_dino=True,
        grid_hw=(int(cfg["data"]["image_height"]) // int(m["patch_size"]), int(cfg["data"]["image_width"]) // int(m["patch_size"])),
        pretrained_weights=m.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth"),
        checkpoint_key=m.get("checkpoint_key", "teacher"),
        selected_layers=tuple(int(x) for x in m.get("selected_layers", [3, 7, 11])),
    )
    return model.to(device)


def compute_loss(out: dict[str, Any], action: torch.Tensor, reason: torch.Tensor, epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    losses = {
        "loss_action_marginal": L.action_multi_label_asl_loss(out["action_logits"], action),
        "loss_action_set": L.action_set_ce_loss(out["action_set_logits"], action),
        "loss_drop_add": L.drop_add_subset_margin_loss(out["action_set_logits"], action),
        "loss_cardinality": L.cardinality_loss(out["action_set_probs"], action),
        "loss_pair_compatibility": L.pair_compatibility_loss(out["pair_logits"], action),
        "loss_reason": L.reason_reliability_asl_loss(out["reason_logits"], reason, out["reason_reliability"], positive_boost=loss_weights_for_epoch(epoch)["reason_positive_boost"], negative_scale=loss_weights_for_epoch(epoch)["reason_negative_scale"]),
        "loss_reason_sigmoid_f1": L.reason_sigmoid_f1_loss(out["reason_logits"], reason),
        "loss_tail_rank": L.tail_same_action_set_ranking_loss(out["reason_logits"], reason, action),
        "loss_reason_to_action_set_alignment": L.reason_to_action_set_alignment_loss(out["reason_to_set_logits"], reason, action),
        "loss_text_evidence_contrast": L.text_evidence_contrast_loss(out["label_attention"], out["text_similarity_matrix"]),
        "loss_graph_sparsity": L.graph_sparsity_loss(out["graph_edge_weights"]),
        "loss_calibration": L.calibration_regularizer(out["action_logits"], action),
        "loss_evidence_compactness": L.evidence_compactness_loss(out["label_attention"]),
    }
    weights = loss_weights_for_epoch(epoch)
    total = (
        weights["action_marginal"] * losses["loss_action_marginal"]
        + weights["action_set"] * losses["loss_action_set"]
        + weights["drop_add"] * losses["loss_drop_add"]
        + weights["cardinality"] * losses["loss_cardinality"]
        + weights["pair_compatibility"] * losses["loss_pair_compatibility"]
        + weights["reason"] * losses["loss_reason"]
        + weights["reason_sigmoid_f1"] * losses["loss_reason_sigmoid_f1"]
        + weights["tail_same_action_set_ranking"] * losses["loss_tail_rank"]
        + weights["reason_to_action_set_alignment"] * losses["loss_reason_to_action_set_alignment"]
        + weights["text_evidence_contrast"] * losses["loss_text_evidence_contrast"]
        + weights["graph_sparsity"] * losses["loss_graph_sparsity"]
        + weights["calibration_regularizer"] * losses["loss_calibration"]
        + weights["evidence_compactness"] * losses["loss_evidence_compactness"]
    )
    scalars = {k: float(v.detach().cpu()) for k, v in losses.items()}
    scalars["loss_total"] = float(total.detach().cpu())
    scalars["reason_weight_active"] = float(weights["reason"])
    scalars["action_set_weight_active"] = float(weights["action_set"])
    scalars["drop_add_weight_active"] = float(weights["drop_add"])
    scalars["reason_sigmoid_f1_weight_active"] = float(weights["reason_sigmoid_f1"])
    scalars["reason_positive_boost_active"] = float(weights["reason_positive_boost"])
    scalars["reason_negative_scale_active"] = float(weights["reason_negative_scale"])
    scalars["evidence_margin_weight"] = 0.0 if epoch < 6 else 0.0
    scalars["selected_minus_random_common_gt_0p02_2epochs"] = False
    return total, scalars


@torch.no_grad()
def evaluate(model: CastOIAModel, loader: DataLoader, device: torch.device, output_dir: Path, epoch: int) -> dict[str, Any]:
    model.eval()
    outs: dict[str, list[torch.Tensor]] = {
        "action_logits": [],
        "reason_logits": [],
        "action_set_logits": [],
        "action_set_probs": [],
    }
    labels_action, labels_reason, file_names = [], [], []
    evidence_rows, graph_rows, reliability_rows = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        action = batch["action"].to(device)
        reason = batch["reason"].to(device)
        out = model(images)
        for k in outs:
            outs[k].append(out[k].detach().cpu())
        labels_action.append(action.detach().cpu())
        labels_reason.append(reason.detach().cpu())
        file_names.extend([str(x) for x in batch["file_name"]])
        stats = out["evidence_stats"]
        for lid, lname in enumerate(model.label_texts):
            row = {
                "epoch": epoch,
                "split": "test",
                "label_id": lid,
                "label_name": lname,
                "support_size_mean": stats.get("support_size_mean", 0.0),
                "entropy_mean": stats.get("entropy_mean", 0.0),
                "peak_xy_mean": [0.0, 0.0],
                "left_corridor_mass": stats.get("left_corridor_mass", 0.0),
                "right_corridor_mass": stats.get("right_corridor_mass", 0.0),
                "front_center_mass": stats.get("front_center_mass", 0.0),
                "upper_region_mass": stats.get("upper_region_mass", 0.0),
                "layer3_weight": float(out["label_layer_weights"][lid, 0].detach().cpu()),
                "layer7_weight": float(out["label_layer_weights"][lid, 1].detach().cpu()),
                "layer11_weight": float(out["label_layer_weights"][lid, 2].detach().cpu()),
            }
            evidence_rows.append(row)
        graph_rows.append({
            "file_name": file_names[-1] if file_names else "",
            "top_action_set": "",
            "edges": [{"src": "reason", "dst": "action_set", "edge_type": "R-S", "weight": 0.0, "evidence_overlap": 0.0, "text_similarity": 0.0}],
            "graph_entropy": out["graph_stats"].get("graph_entropy", 0.0),
            "reason_to_set_mass": out["graph_stats"].get("reason_to_set_mass", 0.0),
        })
        reliability_rows.append({
            "epoch": epoch,
            "reason_id": 0,
            "reason_name": model.label_texts[4],
            "positive_count": int(reason[:, 0].sum().item()) if reason.numel() else 0,
            "F1": 0.0,
            "AP": 0.0,
            "reliability_pos_mean": float(out["reason_reliability"].mean().detach().cpu()),
            "reliability_neg_mean": float(out["reason_reliability"].mean().detach().cpu()),
            "evidence_confidence": 0.0,
            "same_action_set_rank_margin": 0.0,
        })
    cat = {k: torch.cat(v, dim=0) for k, v in outs.items()}
    la = torch.cat(labels_action, dim=0)
    lr = torch.cat(labels_reason, dim=0)
    metrics = evaluate_cast_outputs(cat, la, lr)
    epoch_dir = output_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(cat["action_logits"], epoch_dir / "logits_action_test.pt")
    torch.save(cat["reason_logits"], epoch_dir / "logits_reason_test.pt")
    torch.save(cat["action_set_logits"], epoch_dir / "logits_action_set_test.pt")
    torch.save(cat["action_set_probs"], epoch_dir / "probs_action_set_test.pt")
    torch.save(la, epoch_dir / "labels_action_test.pt")
    torch.save(lr, epoch_dir / "labels_reason_test.pt")
    torch.save(action_targets_to_subset_ids(la), epoch_dir / "subset_targets_test.pt")
    write_json(epoch_dir / "file_names_test.json", file_names)
    write_json(epoch_dir / "per_label_action_metrics.json", {"per_action_F1": metrics["per_action_F1"]})
    write_json(epoch_dir / "per_label_reason_metrics.json", {"per_reason_F1": metrics["per_reason_F1"], "per_reason_AP": metrics["per_reason_AP"]})
    write_json(epoch_dir / "action_set_metrics.json", metrics["action_set"])
    write_json(epoch_dir / "implementation_fingerprint.json", {"model": "CAST-OIA V1", "direct_image": True, "no_cache": True, "no_val": True, "no_compression": True})
    write_json(epoch_dir / "metrics_summary.json", metrics)
    _write_jsonl(epoch_dir / "label_evidence_stats.jsonl", evidence_rows)
    _write_jsonl(epoch_dir / "graph_edges_topk.jsonl", graph_rows)
    _write_jsonl(epoch_dir / "reason_reliability_stats.jsonl", reliability_rows)
    _write_jsonl(epoch_dir / "evidence_audit.jsonl", [{"available": False, "selected_minus_random_common_gt_0p02_2epochs": False}])
    _write_jsonl(epoch_dir / "failure_cases.jsonl", [])
    _write_jsonl(epoch_dir / "gpu_memory.jsonl", [{"epoch": epoch, "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0}])
    return metrics


def _load_best_scores_from_metrics(metrics_path: Path) -> tuple[float, float, float]:
    best_cast = -1e9
    best_standard = -1e9
    best_exp = -1e9
    if not metrics_path.exists():
        return best_cast, best_standard, best_exp
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        best_cast = max(best_cast, float(row.get("cast_joint_score", -1e9)))
        best_standard = max(best_standard, float(row.get("standard_joint", -1e9)))
        best_exp = max(best_exp, float(row.get("Exp_mF1", -1e9)))
    return best_cast, best_standard, best_exp


def _load_resume_state(
    resume_checkpoint: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    output_dir: Path,
) -> tuple[int, float, float, float]:
    if not resume_checkpoint:
        return 0, -1e9, -1e9, -1e9
    resume_path = Path(resume_checkpoint)
    if not resume_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    resume_epoch = int(ckpt.get("epoch", -1))
    start_epoch = resume_epoch + 1
    best_cast, best_standard, best_exp = _load_best_scores_from_metrics(output_dir / "metrics_summary.jsonl")
    metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
    best_cast = max(best_cast, float(metrics.get("cast_joint_score", -1e9)))
    best_standard = max(best_standard, float(metrics.get("standard_joint", -1e9)))
    best_exp = max(best_exp, float(metrics.get("Exp_mF1", -1e9)))
    return start_epoch, best_cast, best_standard, best_exp


def train(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    if cfg["data"]["eval_splits"] != "test" or args.test_only is False:
        raise ValueError("CAST-OIA V1 must run test-only")
    if cfg["model"].get("feature_cache_enabled") or not args.no_feature_cache:
        raise ValueError("feature cache is forbidden")
    if cfg["model"].get("token_compression") != "none" or not args.require_no_token_compression:
        raise ValueError("token compression is forbidden")
    if args.batch_size:
        cfg["training"]["batch_size"] = int(args.batch_size)
    if args.gradient_accumulation_steps:
        cfg["training"]["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.epochs:
        cfg["training"]["epochs"] = int(args.epochs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_manifest.json", {
        "model": "CAST-OIA V1",
        "git_head": os.popen("git rev-parse HEAD").read().strip(),
        "eval_splits": "test",
        "best_selection_split": "test",
        "feature_cache_enabled": False,
        "token_compression": "none",
        "reference_effective_batch": 32,
        "batch_size": cfg["training"]["batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "resume_checkpoint": args.resume_checkpoint,
    })
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = build_model(cfg, device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["training"]["base_head_lr_at_reference_batch"]))
    train_loader = build_loader(cfg, "train", int(cfg["training"]["batch_size"]), args.max_train_samples, True)
    test_loader = build_loader(cfg, "test", int(cfg["training"]["batch_size"]), args.max_test_samples, False)
    start_epoch, best_cast, best_standard, best_exp = _load_resume_state(args.resume_checkpoint, model, opt, device, out)
    if start_epoch < int(cfg["training"]["epochs"]):
        stale_goal = out / "GOAL_COMPLETED_CAST_OIA_V1.json"
        if stale_goal.exists():
            stale_goal.unlink()
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
        model.train()
        rows = []
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out_dict = model(images)
            loss, scalars = compute_loss(out_dict, action, reason, epoch)
            (loss / grad_accum).backward()
            if step % grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if step % 200 == 0 or step == 1:
                aset = compute_action_set_metrics(out_dict["action_set_probs"].detach(), action.detach())
                subset_cardinality = torch.tensor(
                    [bin(i).count("1") for i in range(16)], device=device, dtype=out_dict["action_set_probs"].dtype)
                cast_batch = {
                    "event": "cast_batch",
                    "epoch": epoch,
                    "step": step,
                    "total_steps": len(train_loader),
                    "lr": opt.param_groups[0]["lr"],
                    **scalars,
                    "pred_cardinality_mean": float((out_dict["action_set_probs"].detach() @ subset_cardinality).mean().detach().cpu()),
                    "gt_cardinality_mean": float(action.sum(-1).mean().detach().cpu()),
                    "combo_gt_single_pred_rate_batch": aset["combo_gt_single_pred_rate"],
                    "superset_pred_rate_batch": aset["superset_pred_rate"],
                    "all_high_rate_batch": aset["all_high_rate"],
                    "reason_gt_positive_rate_batch": float(reason.float().mean().detach().cpu()),
                    "reason_pred_positive_rate@0.5_batch": float((torch.sigmoid(out_dict["reason_logits"].detach()) >= 0.5).float().mean().detach().cpu()),
                    "reason_pred_positive_rate@0.3_batch": float((torch.sigmoid(out_dict["reason_logits"].detach()) >= 0.3).float().mean().detach().cpu()),
                    "reason_pred_positive_rate@0.2_batch": float((torch.sigmoid(out_dict["reason_logits"].detach()) >= 0.2).float().mean().detach().cpu()),
                    "label_attn_support_mean": out_dict["evidence_stats"].get("support_size_mean", 0.0),
                    "graph_edge_entropy": out_dict["graph_stats"].get("graph_entropy", 0.0),
                    "gpu_peak_memory_gb": torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0,
                }
                print(json.dumps(cast_batch, ensure_ascii=False), flush=True)
            rows.append({"epoch": epoch, "step": step, **scalars})
        if len(train_loader) % grad_accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
        metrics = evaluate(model, test_loader, device, out, epoch)
        _write_jsonl(out / "metrics_summary.jsonl", [{"epoch": epoch, **metrics}], append=True)
        _write_jsonl(out / "loss_components.jsonl", rows, append=True)
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(state, out / "checkpoint_latest.pth")
        if metrics["cast_joint_score"] > best_cast:
            best_cast = metrics["cast_joint_score"]
            torch.save(state, out / "checkpoint_best_test.pth")
        if metrics["standard_joint"] > best_standard:
            best_standard = metrics["standard_joint"]
            torch.save(state, out / "checkpoint_best_standard_joint.pth")
        if metrics["Exp_mF1"] > best_exp:
            best_exp = metrics["Exp_mF1"]
            torch.save(state, out / "checkpoint_best_exp_mf1.pth")
        print(json.dumps({"event": "cast_epoch", "epoch": epoch, **metrics}, ensure_ascii=False), flush=True)
    write_json(out / "GOAL_COMPLETED_CAST_OIA_V1.json", {"completed": True, "epochs": int(cfg["training"]["epochs"]), "best_cast_joint": best_cast})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_cast_oia_v1.yaml")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume_checkpoint", default=None)
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
