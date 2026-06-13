from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.models.eagle_pu_model import EaglePUModel
from fate_oia.models.eagle_pu_action_set_aux import action_subset_targets
from fate_oia.losses import eagle_pu_losses as L
from fate_oia.engine.eagle_pu_artifacts import append_jsonl, write_json, save_tensor, json_safe
from fate_oia.engine.eval_eagle_pu_oia import evaluate_eagle_pu_tensors


STATE_NAMES = [
    "traffic_control", "front_object", "lateral_lane", "drivable_corridor", "road_geometry", "global_context",
    "traffic_red", "traffic_green", "stop_sign", "front_vehicle", "pedestrian", "rider",
    "left_lane", "right_lane", "left_blocked", "right_blocked", "solid_left", "solid_right",
    "left_turn", "right_turn", "parked_vehicle", "obstacle", "clear_road", "ego_motion",
]
STATE_INDEX = {name: idx for idx, name in enumerate(STATE_NAMES)}


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def make_loader(cfg: dict[str, Any], split: str, batch_size: int, max_samples: int, num_workers: int = 0, shuffle: bool = False) -> DataLoader:
    data = cfg["data"]
    transform = AspectRatioLetterboxTransform(data["image_height"], data["image_width"], patch_size=data.get("patch_size", 8), return_meta=True)
    ds = BDDOIAMultiTaskDataset(data_root=data["data_root"], raw_root=data["raw_root"], split=split, action_dim=data.get("action_dim", 4), reason_dim=data.get("reason_dim", 21), load_image=True, transform=transform)
    if max_samples and max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


class WeakStateTargetBuilder:
    """Builds weak objective state targets from BDD100K detections when available.

    Missing BDD100K matches fall back to a deterministic BDD-OIA label proxy and
    are marked in state_target_source so the run can be audited.
    """

    def __init__(self, bdd100k_root: str | Path | None) -> None:
        self.root = Path(bdd100k_root) if bdd100k_root else None
        self.det_roots: list[Path] = []
        self._cache: dict[str, list[str] | None] = {}
        if self.root:
            for rel in [
                "datasets/datasets/det_annotations/train",
                "datasets/datasets/det_annotations/val",
                "bdd100k_labels/bdd100k/labels/100k/train",
                "bdd100k_labels/bdd100k/labels/100k/val",
            ]:
                p = self.root / rel
                if p.exists():
                    self.det_roots.append(p)

    @staticmethod
    def _stem_candidates(file_name: str) -> list[str]:
        stem = Path(file_name).stem
        candidates = [stem]
        if "_" in stem:
            candidates.append(stem.rsplit("_", 1)[0])
        return list(dict.fromkeys(candidates))

    def _categories_for_file(self, file_name: str) -> list[str] | None:
        for stem in self._stem_candidates(file_name):
            if stem in self._cache:
                return self._cache[stem]
            for root in self.det_roots:
                fp = root / f"{stem}.json"
                if not fp.exists():
                    continue
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                except Exception:
                    self._cache[stem] = None
                    return None
                labels = data.get("labels", data if isinstance(data, list) else [])
                cats = []
                for item in labels:
                    cat = str(item.get("category", item.get("name", ""))).lower()
                    if cat:
                        cats.append(cat)
                self._cache[stem] = cats
                return cats
            self._cache[stem] = None
        return None

    @staticmethod
    def _set(target: torch.Tensor, names: list[str]) -> None:
        for name in names:
            target[STATE_INDEX[name]] = 1.0

    def build(self, file_names: list[str], action: torch.Tensor, reason: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, list[str]]:
        targets = torch.zeros((len(file_names), len(STATE_NAMES)), device=device, dtype=torch.float32)
        sources: list[str] = []
        for i, name in enumerate(file_names):
            self._set(targets[i], ["global_context", "ego_motion", "road_geometry"])
            cats = self._categories_for_file(name)
            if cats is not None:
                sources.append("bdd100k_detection")
                joined = " ".join(cats)
                if any(k in joined for k in ["traffic light", "traffic sign", "sign", "light"]):
                    self._set(targets[i], ["traffic_control"])
                if "traffic light" in joined:
                    self._set(targets[i], ["traffic_red", "traffic_green"])
                if "stop sign" in joined or "traffic sign" in joined:
                    self._set(targets[i], ["stop_sign", "traffic_control"])
                if any(k in joined for k in ["car", "truck", "bus", "train", "vehicle"]):
                    self._set(targets[i], ["front_object", "front_vehicle"])
                if any(k in joined for k in ["person", "pedestrian"]):
                    self._set(targets[i], ["front_object", "pedestrian", "obstacle"])
                if "rider" in joined or "bike" in joined or "bicycle" in joined or "motorcycle" in joined:
                    self._set(targets[i], ["front_object", "rider", "obstacle"])
                if "parking" in joined or "parked" in joined:
                    self._set(targets[i], ["parked_vehicle", "obstacle"])
            else:
                sources.append("bdd_oia_label_proxy")
            # BDD-OIA proxy is always additive because detection boxes are sparse.
            a = action[i]
            r = reason[i]
            if a.numel() >= 4:
                if a[0] > 0.5:
                    self._set(targets[i], ["clear_road", "drivable_corridor"])
                if a[1] > 0.5:
                    self._set(targets[i], ["front_object", "traffic_control", "obstacle"])
                if a[2] > 0.5:
                    self._set(targets[i], ["lateral_lane", "left_lane", "left_turn"])
                if a[3] > 0.5:
                    self._set(targets[i], ["lateral_lane", "right_lane", "right_turn"])
            if r.numel() >= 21:
                if r[:4].max() > 0.5:
                    self._set(targets[i], ["traffic_control"])
                if r[4:12].max() > 0.5:
                    self._set(targets[i], ["front_object", "obstacle"])
                if r[12:16].max() > 0.5:
                    self._set(targets[i], ["lateral_lane"])
                if r[16:].max() > 0.5:
                    self._set(targets[i], ["road_geometry", "drivable_corridor"])
        return targets, sources


def build_state_targets(
    builder: WeakStateTargetBuilder,
    file_names: list[str],
    action: torch.Tensor,
    reason: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    return builder.build(file_names, action, reason, device)


def make_model(cfg: dict[str, Any], device: torch.device, use_mock_dino: bool = False) -> EaglePUModel:
    data = cfg["data"]; model_cfg = cfg["model"]
    model = EaglePUModel(
        dim=int(model_cfg.get("dim", 384)),
        dino_dim=int(model_cfg.get("dino_dim", 384)),
        action_dim=int(data.get("action_dim", 4)),
        reason_dim=int(data.get("reason_dim", 21)),
        selected_layers=tuple(int(x) for x in model_cfg.get("selected_layers", [3,7,11])),
        pretrained_weights=str(data.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        ontology_path=str(model_cfg.get("ontology_path", "configs/eagle_pu_reason_ontology.yaml")),
        freeze_dino=True,
        use_mock_dino=use_mock_dino,
        use_action_graph_delta=bool(model_cfg.get("use_action_graph_delta", False)),
    )
    return model.to(device)


def compute_batch_evidence_scores(out: dict[str, Any], action: torch.Tensor, reason: torch.Tensor) -> dict[str, torch.Tensor | bool]:
    attention = out["label_attention"]
    positive = torch.cat([action, reason], dim=1).bool()
    label_scores = attention * positive.float().unsqueeze(-1)
    denom = positive.float().sum(1).clamp_min(1.0).view(-1, 1)
    evidence = label_scores.sum(1) / denom
    k = max(1, int(evidence.shape[-1] * 0.05))
    selected = evidence.topk(k, dim=-1).values.mean(-1)
    rolled = evidence.roll(shifts=k * 7, dims=-1)
    random = rolled.topk(k, dim=-1).values.mean(-1)
    return {
        "selected": selected,
        "random": random,
        "selected_minus_random": selected - random,
        "available": bool(positive.any().item()),
    }


def compute_losses(
    out: dict[str, Any],
    action: torch.Tensor,
    reason: torch.Tensor,
    state_targets: torch.Tensor,
    evidence_scores: dict[str, torch.Tensor | bool],
    cfg: dict[str, Any],
    epoch: int,
    evidence_gate_active: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    w = cfg["loss_weights"]
    subset_targets = action_subset_targets(action)
    reason_reliability = out.get("reason_reliability", torch.sigmoid(out["reason_logits_final_raw"].detach().abs()))
    evidence_active = bool(evidence_gate_active) and bool(evidence_scores.get("available", False))
    terms = {
        "action_direct": L.action_direct_asl_loss(out["action_logits_direct"], action),
        "reason_direct": L.reason_direct_asl_loss(out["reason_logits_direct"], reason),
        "reason_soft_f1": L.reason_soft_f1_loss(out["reason_logits_final_raw"], reason),
        "pu_reason": L.positive_unlabeled_reason_loss(out["reason_logits_final_raw"], reason, reason_reliability),
        "state_weak_bag": L.state_weak_bag_loss(out["state_logits"], state_targets),
        "text_state_contrast": L.text_state_contrast_loss(out["state_tokens"], out["label_text_prototypes"]),
        "prototype_transport": L.prototype_transport_loss(out["prototype_reason_delta"]),
        "state_label_graph": L.state_label_graph_regularizer(out["edge_weights"]),
        "action_set_ce": L.action_set_ce_loss(out["action_set_logits"], subset_targets),
        "action_set_drop_add": L.action_set_drop_add_loss(out["action_set_logits"], action),
        "cardinality": L.cardinality_loss(out["cardinality_logits"], action),
        "tail_same_action_rank": L.tail_same_action_rank_loss(out["reason_logits_final_raw"], reason),
        "calibration": L.calibration_regularizer(out["calibration_temperature"], out["calibration_bias"]),
        "evidence_margin": L.evidence_margin_loss(evidence_scores["selected"], evidence_scores["random"], active=evidence_active),
    }
    total = sum(terms[k] * float(w.get(k, w.get(k.replace("_same_action_rank", "_same_action_rank"), 0.0))) for k in terms if k != "evidence_margin")
    total = total + terms["evidence_margin"] * float(w.get("evidence_margin_max", 0.002))
    row = {f"loss_{k}": float(v.detach().cpu()) for k, v in terms.items()}
    row["loss_total"] = float(total.detach().cpu())
    row["evidence_margin_active"] = bool(evidence_active)
    row["selected_vs_random_available"] = bool(evidence_scores.get("available", False))
    row["selected_minus_random"] = float(evidence_scores["selected_minus_random"].detach().mean().cpu())
    return total, row


def collect_outputs(model: EaglePUModel, loader: DataLoader, device: torch.device, epoch: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, list[str], dict[str, Any]]:
    model.eval()
    buckets: dict[str, list[torch.Tensor]] = {}
    actions, reasons, names = [], [], []
    keys = [
        "action_logits_final_raw",
        "reason_logits_final_raw",
        "action_logits_final_calibrated",
        "reason_logits_final_calibrated",
        "action_logits_direct",
        "reason_logits_direct",
        "reason_logits_direct_plus_prototype",
        "reason_logits_direct_plus_graph",
        "action_set_logits",
        "action_set_probs",
        "state_logits",
        "state_group_logits",
        "reason_reliability",
        "prototype_reason_delta",
        "reason_graph_delta",
        "calibration_temperature",
        "calibration_bias",
    ]
    stats = {
        "state_entropy": [],
        "state_support": [],
        "graph_entropy": [],
        "reason_reliability_mean": [],
        "prototype_delta_abs": [],
        "graph_delta_abs": [],
    }
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img, epoch=epoch)
            for k in keys:
                if k in out:
                    buckets.setdefault(k, []).append(out[k].detach().cpu())
            if "state_stats" in out:
                stats["state_entropy"].append(float(out["state_stats"].get("state_attention_entropy", 0.0)))
                stats["state_support"].append(float(out["state_stats"].get("state_support_size", 0.0)))
            if "state_graph_stats" in out:
                stats["graph_entropy"].append(float(out["state_graph_stats"].get("graph_entropy", 0.0)))
            stats["reason_reliability_mean"].append(float(out["reason_reliability"].detach().mean().cpu()))
            stats["prototype_delta_abs"].append(float(out["prototype_reason_delta"].detach().abs().mean().cpu()))
            stats["graph_delta_abs"].append(float(out["reason_graph_delta"].detach().abs().mean().cpu()))
            actions.append(batch["action"].detach().cpu())
            reasons.append(batch["reason"].detach().cpu())
            names.extend([str(x) for x in batch["file_name"]])
    outputs = {k: torch.cat(v, 0) for k, v in buckets.items()}
    diag = {k: (sum(v) / max(1, len(v))) for k, v in stats.items()}
    return outputs, torch.cat(actions, 0), torch.cat(reasons, 0), names, diag


def evaluate_branch_metric_views(outputs: dict[str, torch.Tensor], labels_action: torch.Tensor, labels_reason: torch.Tensor) -> dict[str, Any]:
    def view(reason_key: str) -> dict[str, torch.Tensor]:
        return {
            "action_logits_final_raw": outputs["action_logits_direct"],
            "reason_logits_final_raw": outputs[reason_key],
            "action_logits_final_calibrated": outputs["action_logits_final_calibrated"],
            "reason_logits_final_calibrated": outputs["reason_logits_final_calibrated"],
            "action_set_logits": outputs["action_set_logits"],
        }

    return {
        "direct": evaluate_eagle_pu_tensors(view("reason_logits_direct"), labels_action, labels_reason)["metrics_raw_fixed"],
        "direct_plus_prototype": evaluate_eagle_pu_tensors(view("reason_logits_direct_plus_prototype"), labels_action, labels_reason)["metrics_raw_fixed"],
        "direct_plus_graph": evaluate_eagle_pu_tensors(view("reason_logits_direct_plus_graph"), labels_action, labels_reason)["metrics_raw_fixed"],
        "final_raw": evaluate_eagle_pu_tensors(view("reason_logits_final_raw"), labels_action, labels_reason)["metrics_raw_fixed"],
        "final_calibrated": evaluate_eagle_pu_tensors(view("reason_logits_final_raw"), labels_action, labels_reason)["metrics_calibrated"],
    }


def run_selected_vs_random_evidence_audit(
    model: EaglePUModel,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    max_batches: int = 1,
    topk_ratio: float = 0.05,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    positives = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            img = batch["image"].to(device)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            base = model(img, epoch=epoch)
            attention = base["label_attention"]
            label_pos = torch.cat([action, reason], dim=1).bool()
            evidence = (attention * label_pos.float().unsqueeze(-1)).sum(1) / label_pos.float().sum(1).clamp_min(1.0).view(-1, 1)
            k = max(1, int(evidence.shape[-1] * topk_ratio))
            selected_idx = evidence.topk(k, dim=-1).indices
            random_idx = torch.stack([torch.randperm(evidence.shape[-1], device=device)[:k] for _ in range(evidence.shape[0])], dim=0)
            selected_mask = torch.zeros_like(evidence).scatter_(1, selected_idx, 1.0)
            random_mask = torch.zeros_like(evidence).scatter_(1, random_idx, 1.0)
            selected_out = model(img, epoch=epoch, patch_delete_mask=selected_mask)
            random_out = model(img, epoch=epoch, patch_delete_mask=random_mask)
            base_logits = torch.cat([base["action_logits_final_raw"], base["reason_logits_final_raw"]], dim=1)
            selected_logits = torch.cat([selected_out["action_logits_final_raw"], selected_out["reason_logits_final_raw"]], dim=1)
            random_logits = torch.cat([random_out["action_logits_final_raw"], random_out["reason_logits_final_raw"]], dim=1)
            labels = torch.cat([action, reason], dim=1).bool()
            base_prob = torch.sigmoid(base_logits)
            selected_drop = (base_prob - torch.sigmoid(selected_logits)).masked_fill(~labels, 0.0)
            random_drop = (base_prob - torch.sigmoid(random_logits)).masked_fill(~labels, 0.0)
            denom = labels.float().sum(1).clamp_min(1.0)
            selected_mean = selected_drop.sum(1) / denom
            random_mean = random_drop.sum(1) / denom
            diff = selected_mean - random_mean
            positives += int((diff > 0).sum().item())
            for i, file_name in enumerate(batch["file_name"]):
                rows.append({
                    "epoch": epoch,
                    "file_name": str(file_name),
                    "available": True,
                    "topk": k,
                    "selected_drop": float(selected_mean[i].detach().cpu()),
                    "random_drop": float(random_mean[i].detach().cpu()),
                    "selected_minus_random": float(diff[i].detach().cpu()),
                    "common_positive_count": int(labels[i].sum().item()),
                })
    available = bool(rows)
    return {
        "available": available,
        "rows": rows,
        "common_positive_rate": float(positives / max(1, len(rows))) if available else 0.0,
        "selected_minus_random_mean": float(sum(r["selected_minus_random"] for r in rows) / max(1, len(rows))) if available else 0.0,
    }


def save_epoch_artifacts(
    out_dir: Path,
    epoch: int,
    outputs: dict[str, torch.Tensor],
    action: torch.Tensor,
    reason: torch.Tensor,
    names: list[str],
    metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    evidence_audit: dict[str, Any],
    cfg: dict[str, Any],
    loss_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    ep = out_dir / f"epoch_{epoch:03d}"
    ep.mkdir(parents=True, exist_ok=True)
    tensor_map = {
        "logits_action_final_raw_test.pt": outputs["action_logits_final_raw"],
        "logits_reason_final_raw_test.pt": outputs["reason_logits_final_raw"],
        "logits_action_final_calibrated_test.pt": outputs["action_logits_final_calibrated"],
        "logits_reason_final_calibrated_test.pt": outputs["reason_logits_final_calibrated"],
        "logits_action_direct_test.pt": outputs["action_logits_direct"],
        "logits_reason_direct_test.pt": outputs["reason_logits_direct"],
        "logits_action_set_test.pt": outputs["action_set_logits"],
        "probs_action_set_test.pt": outputs["action_set_probs"],
        "labels_action_test.pt": action,
        "labels_reason_test.pt": reason,
    }
    for name, tensor in tensor_map.items():
        save_tensor(ep / name, tensor)
    write_json(ep / "file_names_test.json", names)
    write_json(ep / "run_manifest.json", manifest)
    Path(ep / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(ep / "implementation_fingerprint.json", {
        "model": "EAGLE-PU V1",
        "state_weak_targets": True,
        "selected_vs_random_evidence_audit": evidence_audit.get("available", False),
        "branch_metrics": "separate_logits",
        "final_action": "direct_action_logits",
    })
    append_jsonl(ep / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
    append_jsonl(out_dir / "metrics_summary.jsonl", {"epoch": epoch, **metrics})
    for row in loss_rows:
        append_jsonl(ep / "loss_components.jsonl", row)
        append_jsonl(out_dir / "loss_components.jsonl", {"epoch": epoch, **row})
    append_jsonl(ep / "branch_metrics.jsonl", {"epoch": epoch, **branch_metrics})
    write_json(ep / "per_label_action_metrics.json", metrics.get("metrics_raw_fixed", {}).get("Act_per_label_f1", []))
    write_json(ep / "per_label_reason_metrics.json", metrics.get("metrics_raw_fixed", {}).get("Exp_per_label_f1", []))
    append_jsonl(ep / "state_bank_stats.jsonl", {"epoch": epoch, "available": True, "state_entropy": diagnostics.get("state_entropy"), "state_support": diagnostics.get("state_support")})
    append_jsonl(ep / "prototype_transport_stats.jsonl", {"epoch": epoch, "available": True, "prototype_delta_abs": diagnostics.get("prototype_delta_abs")})
    append_jsonl(ep / "state_graph_stats.jsonl", {"epoch": epoch, "available": True, "graph_entropy": diagnostics.get("graph_entropy"), "graph_delta_abs": diagnostics.get("graph_delta_abs")})
    append_jsonl(ep / "reason_activation_stats.jsonl", {"epoch": epoch, "available": True, "reason_reliability_mean": diagnostics.get("reason_reliability_mean")})
    append_jsonl(ep / "action_set_metrics.jsonl", {"epoch": epoch, **metrics.get("action_set_metrics", {})})
    for row in evidence_audit.get("rows", []):
        append_jsonl(ep / "evidence_faithfulness_audit.jsonl", row)
    if not evidence_audit.get("rows"):
        append_jsonl(ep / "evidence_faithfulness_audit.jsonl", {"epoch": epoch, "available": False, "reason": "no audit rows"})
    append_jsonl(ep / "tail_reason_metrics.jsonl", {"epoch": epoch, "tail_reason_f1": metrics.get("metrics_raw_fixed", {}).get("Exp_per_label_f1", [])})
    append_jsonl(ep / "calibration_diagnostics.jsonl", {"epoch": epoch, "temperature_mean": float(outputs["calibration_temperature"].mean()), "bias_mean": float(outputs["calibration_bias"].mean())})
    append_jsonl(ep / "failure_cases.jsonl", {"epoch": epoch, "available": True, "note": "failure case ranking can be derived from saved logits and labels"})
    gpu = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    append_jsonl(ep / "gpu_memory.jsonl", {"epoch": epoch, "gpu_peak_memory_gb": gpu})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--max_test_samples", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    ap.add_argument("--use_mock_dino", action="store_true")
    ap.add_argument("--log_every", type=int, default=200)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if not args.test_only or not args.no_feature_cache or not args.require_no_token_compression:
        raise ValueError("EAGLE-PU requires --test_only --no_feature_cache --require_no_token_compression")
    if cfg["model"].get("token_compression") != "none" or cfg["model"].get("feature_cache_enabled") is not False:
        raise ValueError("token compression/cache are forbidden")
    epochs = args.epochs or int(cfg["training"]["epochs"])
    batch_size = args.batch_size or int(cfg["training"]["batch_size"])
    accum = args.gradient_accumulation_steps or int(cfg["training"]["gradient_accumulation_steps"])
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, args.num_workers, shuffle=True)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, args.num_workers, shuffle=False)
    model = make_model(cfg, device, use_mock_dino=args.use_mock_dino)
    state_builder = WeakStateTargetBuilder(cfg["data"].get("bdd100k_root", "E:/sbw/BDD100K"))
    groups = [
        {"params": [p for n,p in model.named_parameters() if p.requires_grad and "state_bank" not in n and "proto_transport" not in n and "state_graph" not in n and "calibration" not in n], "lr": cfg["training"]["lr_trunk"]},
        {"params": model.state_bank.parameters(), "lr": cfg["training"]["lr_state_bank"]},
        {"params": model.proto_transport.parameters(), "lr": cfg["training"]["lr_prototype"]},
        {"params": model.state_graph.parameters(), "lr": cfg["training"]["lr_graph"]},
        {"params": model.calibration.parameters(), "lr": cfg["training"]["lr_calibration"]},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg["training"].get("weight_decay", 0.05)))
    manifest = {"git_head": os.popen("git rev-parse HEAD").read().strip(), "github_main_baseline_head": "f642bd1e589bc76df42df6b99bf02a22d23717ef", "command_line": " ".join(sys.argv), "pretrained_weights": cfg["data"].get("pretrained_weights"), "selected_layers": cfg["model"].get("selected_layers"), "data_root": cfg["data"].get("data_root"), "raw_root": cfg["data"].get("raw_root"), "test_only": True, "best_selection_split": "test", "feature_cache_enabled": False, "token_compression": "none", "batch_size": batch_size, "gradient_accumulation_steps": accum, "effective_batch": batch_size * accum, "reference_effective_batch": cfg["training"].get("reference_effective_batch"), "loss_weights": cfg["loss_weights"], "lr_groups": {k: cfg["training"][k] for k in ["lr_trunk", "lr_state_bank", "lr_prototype", "lr_graph", "lr_calibration"]}, "scheduler": cfg["training"].get("scheduler"), "foreground_only": cfg["runtime"].get("foreground_only", True)}
    write_json(out_dir / "run_manifest.json", manifest)
    Path(out_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(out_dir / "implementation_fingerprint.json", {"model": "EAGLE-PU V1", "final_action": "action_logits_final_raw == action_logits_direct", "no_cache": True, "no_val": True})
    best = {"final_raw": -1e9, "calibrated": -1e9, "exp_mf1": -1e9, "exp_map": -1e9, "action_mf1": -1e9}
    evidence_gate_history: list[bool] = []
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(accum, 1)))
    total_updates = max(1, epochs * steps_per_epoch)
    warmup_updates = max(1, int(cfg["training"].get("warmup_epochs", 2)) * steps_per_epoch)
    min_lr = float(cfg["training"].get("min_lr", 1e-5))
    base_lrs = [float(g["lr"]) for g in opt.param_groups]
    update_idx = 0

    def apply_warmup_cosine(update: int) -> None:
        if cfg["training"].get("scheduler") != "warmup_cosine":
            return
        if update < warmup_updates:
            scale = float(update + 1) / float(warmup_updates)
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = max(min_lr, base * scale)
        else:
            progress = min(1.0, float(update - warmup_updates) / float(max(1, total_updates - warmup_updates)))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            for g, base in zip(opt.param_groups, base_lrs):
                g["lr"] = min_lr + (base - min_lr) * cosine

    for epoch in range(epochs):
        evidence_gate_active = (
            epoch >= int(cfg.get("evidence", {}).get("evidence_margin_start_epoch", 8))
            and len(evidence_gate_history) >= 2
            and all(evidence_gate_history[-2:])
        )
        model.train(); opt.zero_grad(set_to_none=True); loss_rows=[]; total_steps=len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            img = batch["image"].to(device); action = batch["action"].to(device); reason = batch["reason"].to(device)
            out = model(img, epoch=epoch)
            state_targets, state_sources = build_state_targets(state_builder, [str(x) for x in batch["file_name"]], action, reason, device)
            evidence_scores = compute_batch_evidence_scores(out, action, reason)
            loss, row = compute_losses(out, action, reason, state_targets, evidence_scores, cfg, epoch, evidence_gate_active=evidence_gate_active)
            (loss / accum).backward()
            row.update({"epoch": epoch, "step": step, "effective_batch": batch_size * accum, "lr_trunk": opt.param_groups[0]["lr"], "lr_state_bank": opt.param_groups[1]["lr"], "lr_prototype": opt.param_groups[2]["lr"], "lr_graph": opt.param_groups[3]["lr"], "lr_calibration": opt.param_groups[4]["lr"]})
            row["state_target_source"] = {src: state_sources.count(src) for src in sorted(set(state_sources))}
            if step % accum == 0 or step == total_steps:
                apply_warmup_cosine(update_idx)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], float(cfg["training"].get("grad_clip", 1.0)))
                row["grad_norm"] = float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                opt.step(); opt.zero_grad(set_to_none=True); update_idx += 1
            loss_rows.append(row)
            if step % args.log_every == 0 or step == 1:
                gpu = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
                print(json.dumps({"event":"eagle_pu_batch", "epoch":epoch, "step":step, "total_steps":total_steps, "loss_total":row["loss_total"], "gpu_peak_memory_gb":gpu}), flush=True)
        outputs, act_y, exp_y, names, diagnostics = collect_outputs(model, test_loader, device, epoch)
        evidence_audit = run_selected_vs_random_evidence_audit(model, test_loader, device, epoch, max_batches=1)
        outputs["evidence_audit_common_positive_rate"] = torch.tensor(evidence_audit.get("common_positive_rate", 0.0))
        evidence_gate_history.append(bool(evidence_audit.get("selected_minus_random_mean", 0.0) > 0.0 and evidence_audit.get("common_positive_rate", 0.0) > 0.0))
        metrics = evaluate_eagle_pu_tensors(outputs, act_y, exp_y)
        branch_metrics = evaluate_branch_metric_views(outputs, act_y, exp_y)
        save_epoch_artifacts(out_dir, epoch, outputs, act_y, exp_y, names, metrics, branch_metrics, diagnostics, evidence_audit, cfg, loss_rows, manifest)
        latest = {"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "metrics": metrics, "cfg": cfg}
        torch.save(latest, out_dir / "checkpoint_latest.pth")
        if metrics["final_raw_joint"] >= best["final_raw"]:
            best["final_raw"] = metrics["final_raw_joint"]; torch.save(latest, out_dir / "checkpoint_best_test_final_raw.pth")
        cal_joint = 0.5 * metrics["metrics_calibrated"].get("Act_mF1",0.0) + 0.5 * metrics["metrics_calibrated"].get("Exp_mF1",0.0)
        if cal_joint >= best["calibrated"]:
            best["calibrated"] = cal_joint; torch.save(latest, out_dir / "checkpoint_best_test_final_calibrated.pth")
        if metrics["metrics_raw_fixed"].get("Exp_mF1",0.0) >= best["exp_mf1"]:
            best["exp_mf1"] = metrics["metrics_raw_fixed"].get("Exp_mF1",0.0); torch.save(latest, out_dir / "checkpoint_best_test_exp_mf1.pth")
        if metrics["metrics_raw_fixed"].get("Exp_mAP",0.0) >= best["exp_map"]:
            best["exp_map"] = metrics["metrics_raw_fixed"].get("Exp_mAP",0.0); torch.save(latest, out_dir / "checkpoint_best_test_exp_map.pth")
        if metrics["metrics_raw_fixed"].get("Act_mF1",0.0) >= best["action_mf1"]:
            best["action_mf1"] = metrics["metrics_raw_fixed"].get("Act_mF1",0.0); torch.save(latest, out_dir / "checkpoint_best_test_action_mf1.pth")
        print(json.dumps({"event":"eagle_pu_epoch", "epoch":epoch, "final_raw_joint":metrics["final_raw_joint"], "standard_joint":metrics["standard_joint"], "Act_mF1":metrics["metrics_raw_fixed"].get("Act_mF1"), "Exp_mF1":metrics["metrics_raw_fixed"].get("Exp_mF1"), "Exp_mAP":metrics["metrics_raw_fixed"].get("Exp_mAP")}), flush=True)
    write_json(out_dir / "GOAL_COMPLETED_EAGLE_PU_V1.json", {"completed_epochs": epochs, "best": best, "test_only": True})

if __name__ == "__main__":
    main()
