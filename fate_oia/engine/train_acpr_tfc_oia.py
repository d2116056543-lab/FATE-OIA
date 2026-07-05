from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.metrics import multilabel_metrics_from_logits
from fate_oia.models.acpr_tfc_model import ACPRTFCModel
from fate_oia.transforms import AspectRatioLetterboxTransform
from fate_oia.losses.tfc_losses import action_asl_loss, calalign_softf1_loss, compute_tfc_losses, reason_pu_asl_loss, threshold_smooth_loss
from fate_oia.utils.acpr_train_calib_split import make_train_calib_indices


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


REQUIRED_PRETRAIN_GATES = [
    "TFC_GATE_A_CODE_AUDIT_PASS.json",
    "TFC_GATE_B_NO_TEST_LEAKAGE_PASS.json",
    "TFC_GATE_C_ACTION_FIREWALL_PASS.json",
    "TFC_GATE_D_FACTOR_GROUNDING_PASS.json",
    "TFC_GATE_E_SELECTED_DELETION_GT_RANDOM_PASS.json",
    "TFC_GATE_F_PU_STATE_PASS.json",
    "TFC_GATE_G_CALALIGN_PASS.json",
    "TFC_GATE_H_MEMORY_PROBE_PASS.json",
    "acpr_tfc_v1_REVIEW_PASS.json",
]


def missing_pretrain_gates(review_dir: Path = Path(".review")) -> list[str]:
    return [name for name in REQUIRED_PRETRAIN_GATES if not (review_dir / name).exists()]


def pretrain_gate_failures(review_dir: Path = Path(".review")) -> list[str]:
    failures: list[str] = []
    for name in REQUIRED_PRETRAIN_GATES:
        path = review_dir / name
        if not path.exists():
            failures.append(f"{name}:missing")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - defensive for corrupt gate artifacts.
            failures.append(f"{name}:invalid_json:{exc}")
            continue
        key = "review_pass" if name == "acpr_tfc_v1_REVIEW_PASS.json" else "pass"
        if not bool(data.get(key, False)):
            failures.append(f"{name}:{key}=false")
    return failures


def enforce_pretrain_gates(allow_failed_gates: bool, require_review_pass: bool) -> list[str]:
    """Full training is gate-first by default; smoke/debug must opt out explicitly."""
    review_dir = Path(".review")
    if allow_failed_gates:
        if require_review_pass and not (review_dir / "acpr_tfc_v1_REVIEW_PASS.json").exists():
            raise FileNotFoundError(".review/acpr_tfc_v1_REVIEW_PASS.json missing")
        return pretrain_gate_failures(review_dir)
    failures = pretrain_gate_failures(review_dir)
    if failures:
        raise FileNotFoundError(
            "Missing or failed required TFC pretrain gates: "
            + ", ".join(failures)
            + ". Run audit_tfc_gates first or pass --allow_failed_gates only for targeted smoke/debug."
        )
    return []


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "reason": torch.stack([b["reason"] for b in batch]),
        "file_name": [b["file_name"] for b in batch],
    }


def make_loader(cfg: dict, split: str, batch_size: int, max_samples: int | None, shuffle: bool, num_workers: int, indices: list[int] | None = None) -> DataLoader:
    t = AspectRatioLetterboxTransform(cfg.get("image_height", 360), cfg.get("image_width", 640), cfg.get("patch_size", 8))
    ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg.get("raw_root"), split=split, load_image=True, transform=t)
    if indices is not None:
        ds = Subset(ds, indices)
    elif max_samples:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate,
        "pin_memory": bool(cfg.get("training", {}).get("pin_memory", True)) and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("training", {}).get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.get("training", {}).get("prefetch_factor", 4))
    return DataLoader(ds, **kwargs)


def build_model(cfg: dict, device: torch.device) -> ACPRTFCModel:
    model_cfg = cfg.get("model", {})
    tfc = cfg.get("tfc", {})
    return ACPRTFCModel(
        dim=int(model_cfg.get("dim", 384)),
        selected_layers=tuple(model_cfg.get("selected_layers", [3, 7, 11])),
        pretrained_weights=str(cfg.get("pretrained_weights", "ckp/reference/dino_deitsmall8_pretrain.pth")),
        factor_bank_path=str(tfc.get("factor_bank", "configs/acpr_tfc_factors.yaml")),
        factor_topk_tokens=int(tfc.get("factor_topk_tokens", 64)),
        num_factor_prototypes=int(tfc.get("num_factor_prototypes", 4)),
        use_mock_dino=bool(model_cfg.get("use_mock_dino", False)),
        action_delta_max=float(tfc.get("action_delta_max", 0.06)),
        reason_delta_max=float(tfc.get("reason_delta_max", 0.15)),
    ).to(device)


def set_trainable(model: ACPRTFCModel, base_requires: dict[str, bool], mode: str) -> None:
    for name, param in model.named_parameters():
        if mode == "main":
            param.requires_grad_(base_requires[name] and not name.startswith("calalign."))
        elif mode == "calalign":
            param.requires_grad_(base_requires[name] and name.startswith("calalign."))
        else:
            param.requires_grad_(base_requires[name])


def _group_for_param(name: str) -> str:
    if name.startswith("action_head.") or name.startswith("lane_adapter.action_adapter."):
        return "action"
    if name.startswith("reason_head.") or name.startswith("lane_adapter.reason_adapter.") or name.startswith("pu_state."):
        return "reason"
    if name.startswith("prototype_bank.") or name.startswith("measure_action.") or name.startswith("measure_reason."):
        return "factor"
    if name.startswith("target_credit."):
        return "credit"
    return "action"


def build_main_param_groups(model: ACPRTFCModel, train_cfg: dict) -> tuple[list[dict], dict[str, float]]:
    lr_by_group = {
        "action": float(train_cfg.get("lr_action", 2e-4)),
        "reason": float(train_cfg.get("lr_reason", train_cfg.get("lr_action", 2e-4))),
        "factor": float(train_cfg.get("lr_factor", train_cfg.get("lr_action", 2e-4))),
        "credit": float(train_cfg.get("lr_credit", train_cfg.get("lr_action", 2e-4))),
    }
    grouped: dict[str, list[torch.nn.Parameter]] = {name: [] for name in lr_by_group}
    for name, param in model.named_parameters():
        if not param.requires_grad or name.startswith("calalign."):
            continue
        grouped[_group_for_param(name)].append(param)
    param_groups = [
        {"params": params, "lr": lr_by_group[group_name], "initial_lr": lr_by_group[group_name], "group_name": group_name}
        for group_name, params in grouped.items()
        if params
    ]
    return param_groups, lr_by_group


def warmup_cosine_scale(progress_epoch: float, total_epochs: int, warmup_epochs: float, min_lr_ratio: float) -> float:
    if total_epochs <= 0:
        return 1.0
    warmup_epochs = max(float(warmup_epochs), 0.0)
    min_lr_ratio = float(min(max(min_lr_ratio, 0.0), 1.0))
    if warmup_epochs > 0 and progress_epoch < warmup_epochs:
        return max(progress_epoch / warmup_epochs, 1.0 / max(total_epochs, 1))
    denom = max(float(total_epochs) - warmup_epochs, 1e-8)
    cosine_progress = min(max((progress_epoch - warmup_epochs) / denom, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * cosine_progress))


def apply_lr_schedule(optimizer: torch.optim.Optimizer, progress_epoch: float, total_epochs: int, train_cfg: dict) -> dict[str, float]:
    scheduler = str(train_cfg.get("scheduler", "cosine")).lower()
    if scheduler not in {"cosine", "warmup_cosine", "warmup_cosine_by_update"}:
        return {str(group.get("group_name", idx)): float(group["lr"]) for idx, group in enumerate(optimizer.param_groups)}
    scale = warmup_cosine_scale(
        progress_epoch,
        total_epochs,
        float(train_cfg.get("warmup_epochs", 2)),
        float(train_cfg.get("min_lr_ratio", 0.05)),
    )
    current: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        base_lr = float(group.get("initial_lr", group["lr"]))
        group["lr"] = base_lr * scale
        current[str(group.get("group_name", idx))] = float(group["lr"])
    return current


def thresholded_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor, prefix: str) -> dict:
    probs = torch.sigmoid(logits.float())
    labels_bool = labels.float() > 0.5
    preds = probs >= thresholds.view(1, -1)
    tp = (preds & labels_bool).sum(0).float()
    fp = (preds & (~labels_bool)).sum(0).float()
    fn = ((~preds) & labels_bool).sum(0).float()
    per_f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1e-8)).tolist()
    micro_tp = tp.sum()
    micro_fp = fp.sum()
    micro_fn = fn.sum()
    return {
        f"{prefix}mF1": float(torch.tensor(per_f1).mean()),
        f"{prefix}oF1": float(2 * micro_tp / (2 * micro_tp + micro_fp + micro_fn).clamp_min(1e-8)),
        f"{prefix}per_label_f1": per_f1,
        f"{prefix}thresholds": [float(x) for x in thresholds.detach().cpu()],
    }


def oracle_threshold_metrics(logits: torch.Tensor, labels: torch.Tensor, prefix: str) -> dict:
    candidates = torch.linspace(0.05, 0.95, 19, device=logits.device)
    probs = torch.sigmoid(logits.float())
    thresholds = []
    for label_idx in range(labels.shape[1]):
        best_threshold = candidates[0]
        best_f1 = logits.new_tensor(-1.0)
        y = labels[:, label_idx].float() > 0.5
        for threshold in candidates:
            pred = probs[:, label_idx] >= threshold
            tp = (pred & y).sum().float()
            fp = (pred & (~y)).sum().float()
            fn = ((~pred) & y).sum().float()
            f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        thresholds.append(best_threshold)
    return thresholded_metrics_from_logits(logits, labels, torch.stack(thresholds), prefix)


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    positives = y.sum()
    if float(positives) <= 0:
        return 0.0
    ranks = torch.arange(1, y.numel() + 1, device=scores.device, dtype=torch.float32)
    precision_at_k = torch.cumsum(y, dim=0) / ranks
    return float((precision_at_k * y).sum().div(positives).cpu())


def _auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    y = labels.float()
    positives = y.sum()
    negatives = y.numel() - positives
    if float(positives) <= 0 or float(negatives) <= 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores, dtype=torch.float32)
    ranks[order] = torch.arange(1, scores.numel() + 1, device=scores.device, dtype=torch.float32)
    pos_rank_sum = (ranks * y).sum()
    auc = (pos_rank_sum - positives * (positives + 1) / 2) / (positives * negatives).clamp_min(1e-8)
    return float(auc.cpu())


def per_label_ranking_metrics(logits: torch.Tensor, labels: torch.Tensor, f1_values: list[float]) -> list[dict]:
    probs = torch.sigmoid(logits.float())
    labels_bool = labels.float() > 0.5
    rows: list[dict] = []
    for label_idx in range(labels.shape[1]):
        rows.append({
            "label_id": label_idx,
            "F1": float(f1_values[label_idx]) if label_idx < len(f1_values) else None,
            "AP": _average_precision(probs[:, label_idx], labels_bool[:, label_idx]),
            "AUC": _auc_roc(probs[:, label_idx], labels_bool[:, label_idx]),
        })
    return rows


def build_flip_cases(
    epoch: int,
    file_names: list[str],
    visual_logits: torch.Tensor,
    final_logits: torch.Tensor,
    labels: torch.Tensor,
    max_cases: int = 200,
) -> list[dict]:
    visual_pred = torch.sigmoid(visual_logits) >= 0.5
    final_pred = torch.sigmoid(final_logits) >= 0.5
    labels_bool = labels > 0.5
    cases: list[dict] = []
    for sample_idx, file_name in enumerate(file_names):
        for action_id in range(labels.shape[1]):
            before = bool(visual_pred[sample_idx, action_id])
            after = bool(final_pred[sample_idx, action_id])
            gt = bool(labels_bool[sample_idx, action_id])
            if before == after:
                continue
            if (not before) and after and gt:
                transition = "FP_to_TP"
            elif before and (not after) and gt:
                transition = "TP_to_FN"
            elif (not before) and after and (not gt):
                transition = "TN_to_FP"
            else:
                transition = "FN_to_TN"
            cases.append({
                "epoch": epoch,
                "file_name": file_name,
                "action_id": action_id,
                "transition": transition,
                "gt": gt,
                "visual_prob": float(torch.sigmoid(visual_logits[sample_idx, action_id]).cpu()),
                "final_prob": float(torch.sigmoid(final_logits[sample_idx, action_id]).cpu()),
            })
            if len(cases) >= max_cases:
                return cases
    return cases


def _grad_abs_sum(module: torch.nn.Module) -> float:
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().sum().cpu())
    return total


def firewall_gradient_probe(model: ACPRTFCModel, batch: dict, device: torch.device, epoch: int) -> dict:
    """Probe the structural firewall without stepping optimizers."""
    was_training = model.training
    model.train()
    img = batch["image"].to(device, non_blocking=True)
    action = batch["action"].to(device)
    reason = batch["reason"].to(device)

    model.zero_grad(set_to_none=True)
    out_reason = model(img, action, reason, epoch=epoch, split="train", run_deletion=False)
    reason_loss = reason_pu_asl_loss(out_reason["reason_logits_deploy"], reason, out_reason["pu_state"])
    reason_loss.backward()
    reason_loss_action_adapter_grad = _grad_abs_sum(model.lane_adapter.action_adapter)
    reason_loss_action_head_grad = _grad_abs_sum(model.action_head)

    model.zero_grad(set_to_none=True)
    out_action = model(img, action, reason, epoch=epoch, split="train", run_deletion=False)
    action_loss = action_asl_loss(out_action["action_logits_deploy"], action)
    action_loss.backward()
    action_loss_reason_adapter_grad = _grad_abs_sum(model.lane_adapter.reason_adapter)
    action_loss_reason_head_grad = _grad_abs_sum(model.reason_head)
    model.zero_grad(set_to_none=True)
    model.train(was_training)

    return {
        "epoch": epoch,
        "enabled": "structural_firewall",
        "cosine_action_reason": None,
        "cosine_action_factor": None,
        "cosine_action_credit": None,
        "projection_count": 0,
        "reason_loss_action_adapter_grad": reason_loss_action_adapter_grad,
        "reason_loss_action_head_grad": reason_loss_action_head_grad,
        "action_loss_reason_adapter_grad": action_loss_reason_adapter_grad,
        "action_loss_reason_head_grad": action_loss_reason_head_grad,
        "firewall_pass": (
            reason_loss_action_adapter_grad == 0.0
            and reason_loss_action_head_grad == 0.0
            and action_loss_reason_adapter_grad == 0.0
            and action_loss_reason_head_grad == 0.0
        ),
    }


def build_target_credit_rows(epoch: int, out: dict, action_targets: torch.Tensor, reason_targets: torch.Tensor) -> list[dict]:
    rows: list[dict] = []
    comp = out["compatibility"]
    credit_action = out["credit_action"].detach()
    credit_reason = out["credit_reason"].detach()
    native_action = (comp["factor_to_action_support"] - comp["factor_to_action_inhibit"]).to(credit_action.device)
    native_reason = (comp["factor_to_reason_support"] - comp["factor_to_reason_inhibit"]).to(credit_reason.device)
    selected_effect_action = out["deletion_stats_action"]["selected_effect"].detach()
    random_effect_action = out["deletion_stats_action"]["random_effect"].detach()
    gap_action = out["deletion_stats_action"]["selected_vs_random_gap"].detach()
    selected_effect_reason = out["deletion_stats_reason"]["selected_effect"].detach()
    random_effect_reason = out["deletion_stats_reason"]["random_effect"].detach()
    gap_reason = out["deletion_stats_reason"]["selected_vs_random_gap"].detach()

    def sign_acc(values: torch.Tensor, labels: torch.Tensor, compat: torch.Tensor, target_id: int) -> tuple:
        pos_mask = labels[:, target_id] > 0.5
        support = compat > 0
        inhibit = compat < 0
        pos_acc = None
        inh_acc = None
        if bool(pos_mask.any()) and bool(support.item()):
            pos_acc = float((values[pos_mask] > 0).float().mean().cpu())
        if bool(pos_mask.any()) and bool(inhibit.item()):
            inh_acc = float((values[pos_mask] < 0).float().mean().cpu())
        return pos_acc, inh_acc

    for target_id in range(action_targets.shape[1]):
        mean_credit = credit_action[:, :, target_id].mean(0)
        factor_id = int(mean_credit.abs().argmax().cpu())
        pos_acc, inh_acc = sign_acc(credit_action[:, factor_id, target_id], action_targets, native_action[factor_id, target_id], target_id)
        rows.append({
            "epoch": epoch,
            "target_type": "action",
            "target_id": target_id,
            "factor_id": factor_id,
            "credit_mean": float(mean_credit[factor_id].cpu()),
            "credit_topk": float(credit_action[:, factor_id, target_id].abs().mean().cpu()),
            "compatibility": float(native_action[factor_id, target_id].cpu()),
            "deletion_selected": float(selected_effect_action[:, target_id].mean().cpu()),
            "deletion_random": float(random_effect_action[:, target_id].mean().cpu()),
            "selected_vs_random_gap": float(gap_action[:, target_id].mean().cpu()),
            "deletion_available": True,
            "positive_credit_sign_acc": pos_acc,
            "inhibitory_credit_sign_acc": inh_acc,
        })
    for target_id in range(reason_targets.shape[1]):
        mean_credit = credit_reason[:, :, target_id].mean(0)
        factor_id = int(mean_credit.abs().argmax().cpu())
        pos_acc, inh_acc = sign_acc(credit_reason[:, factor_id, target_id], reason_targets, native_reason[factor_id, target_id], target_id)
        rows.append({
            "epoch": epoch,
            "target_type": "reason",
            "target_id": target_id,
            "factor_id": factor_id,
            "credit_mean": float(mean_credit[factor_id].cpu()),
            "credit_topk": float(credit_reason[:, factor_id, target_id].abs().mean().cpu()),
            "compatibility": float(native_reason[factor_id, target_id].cpu()),
            "deletion_selected": float(selected_effect_reason[:, target_id].mean().cpu()),
            "deletion_random": float(random_effect_reason[:, target_id].mean().cpu()),
            "selected_vs_random_gap": float(gap_reason[:, target_id].mean().cpu()),
            "deletion_available": True,
            "positive_credit_sign_acc": pos_acc,
            "inhibitory_credit_sign_acc": inh_acc,
        })
    return rows


@torch.no_grad()
def evaluate(model: ACPRTFCModel, loader: DataLoader, device: torch.device, epoch: int, out_dir: Path) -> dict:
    model.eval()
    act_logits = []; rea_logits = []; act_labels = []; rea_labels = []; names = []
    act_visual = []; act_delta_off = []; act_base = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        a = batch["action"].to(device)
        r = batch["reason"].to(device)
        out = model(img, None, None, epoch=epoch, split="test", run_deletion=True)
        act_logits.append(out["action_logits_deploy"].cpu())
        rea_logits.append(out["reason_logits_deploy"].cpu())
        act_visual.append(out["action_visual_logits"].cpu())
        act_delta_off.append((out["action_logits_base"] - out["action_tfc_delta"]).cpu())
        act_base.append(out["action_logits_base"].cpu())
        act_labels.append(a.cpu()); rea_labels.append(r.cpu()); names.extend(batch["file_name"])
    action_logits = torch.cat(act_logits); reason_logits = torch.cat(rea_logits)
    action_labels = torch.cat(act_labels); reason_labels = torch.cat(rea_labels)
    action_metrics = multilabel_metrics_from_logits(action_logits, action_labels, prefix="Act_")
    reason_metrics = multilabel_metrics_from_logits(reason_logits, reason_labels, prefix="Exp_")
    action_oracle = oracle_threshold_metrics(action_logits, action_labels, prefix="Act_")
    reason_oracle = oracle_threshold_metrics(reason_logits, reason_labels, prefix="Exp_")
    metrics = {**action_metrics, **reason_metrics}
    metrics["Act_oracle_mF1"] = action_oracle["Act_mF1"]
    metrics["Exp_oracle_mF1"] = reason_oracle["Exp_mF1"]
    metrics["joint"] = 0.5 * metrics["Act_mF1"] + 0.5 * metrics["Exp_mF1"]
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    write_json(epoch_dir / "metrics_summary.json", metrics)
    torch.save(action_logits, epoch_dir / "logits_action_deploy_test.pt")
    torch.save(reason_logits, epoch_dir / "logits_reason_deploy_test.pt")
    torch.save(action_labels, epoch_dir / "labels_action_test.pt")
    torch.save(reason_labels, epoch_dir / "labels_reason_test.pt")
    write_json(epoch_dir / "file_names_test.json", names)
    action_visual_logits = torch.cat(act_visual)
    action_delta_off_logits = torch.cat(act_delta_off)
    action_base_logits = torch.cat(act_base)
    visual_pred = torch.sigmoid(action_visual_logits) >= 0.5
    final_pred = torch.sigmoid(action_logits) >= 0.5
    labels_bool = action_labels > 0.5
    fp_to_tp = ((~visual_pred) & final_pred & labels_bool).sum(0).to(torch.int64).tolist()
    tp_to_fn = (visual_pred & (~final_pred) & labels_bool).sum(0).to(torch.int64).tolist()
    tn_to_fp = ((~visual_pred) & final_pred & (~labels_bool)).sum(0).to(torch.int64).tolist()
    fn_to_tn = (visual_pred & (~final_pred) & (~labels_bool)).sum(0).to(torch.int64).tolist()
    action_ranking = per_label_ranking_metrics(action_logits, action_labels, action_metrics.get("Act_per_label_f1", []))
    flip_cases = build_flip_cases(epoch, names, action_visual_logits, action_logits, action_labels)
    write_json(epoch_dir / "action_branch_metrics.json", {
        "action_visual_only": multilabel_metrics_from_logits(action_visual_logits, action_labels, prefix="Act_"),
        "action_tfc_delta_off": multilabel_metrics_from_logits(action_delta_off_logits, action_labels, prefix="Act_"),
        "action_tfc_delta_on": multilabel_metrics_from_logits(action_base_logits, action_labels, prefix="Act_"),
        "action_threshold_delta_off": multilabel_metrics_from_logits(action_base_logits, action_labels, prefix="Act_"),
        "action_final_deploy": action_metrics,
        "action_oracle": action_oracle,
        "per_action_AP_AUC_F1": action_ranking,
        "FP_to_TP": fp_to_tp,
        "TP_to_FN": tp_to_fn,
        "TN_to_FP": tn_to_fp,
        "FN_to_TN": fn_to_tn,
    })
    append_jsonl(out_dir / "failure_flip_cases.jsonl", {"epoch": epoch, "cases": flip_cases})
    append_jsonl(epoch_dir / "failure_flip_cases.jsonl", {"epoch": epoch, "cases": flip_cases})
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_tfc_v1.yaml")
    ap.add_argument("--output_dir", default=".background_runs/acpr_tfc_v1_full")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--max_test_samples", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--require_review_pass", action="store_true")
    ap.add_argument("--allow_failed_gates", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    missing_gates_at_launch = enforce_pretrain_gates(args.allow_failed_gates, args.require_review_pass)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    epochs = int(args.epochs or train_cfg.get("epochs", 14))
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 4))
    accum = int(args.gradient_accumulation_steps or train_cfg.get("grad_accumulation_steps", 8))
    workers = int(args.num_workers if args.num_workers is not None else train_cfg.get("num_workers", 4))
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = BDDOIAMultiTaskDataset(cfg["data_root"], cfg.get("raw_root"), split="train")
    main_idx, calib_idx = make_train_calib_indices(train_ds, calib_fraction=0.10)
    train_loader = make_loader(cfg, "train", batch_size, args.max_train_samples, True, workers, indices=main_idx if args.max_train_samples is None else None)
    train_calib_loader = make_loader(cfg, "train", batch_size, None, True, workers, indices=calib_idx)
    test_loader = make_loader(cfg, "test", batch_size, args.max_test_samples, False, workers)
    model = build_model(cfg, device)
    weights = cfg.get("loss_weights", {})
    base_requires = {name: param.requires_grad for name, param in model.named_parameters()}
    main_param_groups, main_lr_by_group = build_main_param_groups(model, train_cfg)
    main_params = [param for group in main_param_groups for param in group["params"]]
    calalign_params = [param for name, param in model.named_parameters() if param.requires_grad and name.startswith("calalign.")]
    optimizer = torch.optim.AdamW(main_param_groups, weight_decay=float(train_cfg.get("weight_decay", 0.05)))
    threshold_lr = float(cfg.get("threshold", {}).get("lr_threshold", train_cfg.get("lr_threshold", 7e-4)))
    threshold_optimizer = torch.optim.AdamW([{"params": calalign_params, "lr": threshold_lr, "initial_lr": threshold_lr, "group_name": "threshold"}], weight_decay=0.0)
    write_json(out_dir / "run_manifest.json", {
        "config": args.config,
        "data_root": cfg["data_root"],
        "raw_root": cfg.get("raw_root"),
        "test_only_eval": True,
        "feature_cache_enabled": False,
        "token_compression": False,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "num_workers": workers,
        "train_calib_count": len(calib_idx),
        "train_calib_threshold_only": True,
        "main_optimizer_excludes_calalign": True,
        "lr_groups": main_lr_by_group,
        "lr_threshold": threshold_lr,
        "scheduler": train_cfg.get("scheduler", "cosine"),
        "warmup_epochs": train_cfg.get("warmup_epochs", 2),
        "min_lr_ratio": train_cfg.get("min_lr_ratio", 0.05),
        "pretrain_gates_required": not args.allow_failed_gates,
        "allow_failed_gates": bool(args.allow_failed_gates),
        "required_pretrain_gates": REQUIRED_PRETRAIN_GATES,
        "missing_gates_at_launch": missing_gates_at_launch,
        "gate_failures_at_launch": missing_gates_at_launch,
    })
    best_joint = -1.0
    best_action_mf1 = -1.0
    best_exp_mf1 = -1.0
    last_loss_row = {}
    last_train_stats = {}
    qrho_collapse_epochs = 0
    oracle_act_drop_epochs = 0
    prev_oracle_act_mf1: float | None = None
    prev_act_map: float | None = None
    prev_exp_mf1: float | None = None
    prev_exp_deploy_oracle_gap: float | None = None
    steps_per_epoch = max(1, len(train_loader))
    for epoch in range(epochs):
        model.train()
        set_trainable(model, base_requires, "main")
        optimizer.zero_grad(set_to_none=True)
        epoch_deletion_summary: dict | None = None
        epoch_credit_rows: list[dict] = []
        probe_batch: dict | None = None
        for step, batch in enumerate(train_loader):
            if probe_batch is None:
                probe_batch = {
                    "image": batch["image"][: min(2, batch["image"].shape[0])].detach().clone(),
                    "action": batch["action"][: min(2, batch["action"].shape[0])].detach().clone(),
                    "reason": batch["reason"][: min(2, batch["reason"].shape[0])].detach().clone(),
                }
            img = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out = model(img, action, reason, epoch=epoch, split="train", run_deletion=(epoch >= 3 and step % 10 == 0))
            losses = compute_tfc_losses(out, action, reason, weights)
            loss = losses["total"] / accum
            if not torch.isfinite(loss):
                write_json(out_dir / "run_stop_reason.json", {"reason": "nan_or_inf_loss", "epoch": epoch, "step": step})
                raise RuntimeError("NaN/Inf TFC loss")
            loss.backward()
            if (step + 1) % accum == 0:
                current_lrs = apply_lr_schedule(optimizer, epoch + (step + 1) / steps_per_epoch, epochs, train_cfg)
                torch.nn.utils.clip_grad_norm_(main_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                current_lrs = {str(group.get("group_name", idx)): float(group["lr"]) for idx, group in enumerate(optimizer.param_groups)}
            row = {k: float(v.detach().cpu()) for k, v in losses.items() if torch.is_tensor(v)}
            row.update({
                "epoch": epoch,
                "step": step,
                "lr": optimizer.param_groups[0]["lr"],
                "lr_action": current_lrs.get("action"),
                "lr_reason": current_lrs.get("reason"),
                "lr_factor": current_lrs.get("factor"),
                "lr_credit": current_lrs.get("credit"),
            })
            last_loss_row = row
            last_train_stats = {
                "factor_action_prob_mean": float(out["factor_probs_action"].detach().mean().cpu()),
                "factor_reason_prob_mean": float(out["factor_probs_reason"].detach().mean().cpu()),
                "factor_action_rho_mean": float(out["factor_rho_action"].detach().mean().cpu()),
                "factor_reason_rho_mean": float(out["factor_rho_reason"].detach().mean().cpu()),
                "credit_action_abs_mean": float(out["credit_action"].detach().abs().mean().cpu()),
                "credit_reason_abs_mean": float(out["credit_reason"].detach().abs().mean().cpu()),
                "deletion_gap_mean_action": float(out["deletion_stats_action"]["selected_vs_random_gap"].detach().mean().cpu()),
                "deletion_gap_mean_reason": float(out["deletion_stats_reason"]["selected_vs_random_gap"].detach().mean().cpu()),
                "deletion_gap_mean": float(0.5 * (out["deletion_stats_action"]["selected_vs_random_gap"].detach().mean() + out["deletion_stats_reason"]["selected_vs_random_gap"].detach().mean()).cpu()),
                "deletion_selected_gt_random_rate_action": float(out["deletion_stats_action"]["selected_gt_random_rate"].detach().cpu()),
                "deletion_selected_gt_random_rate_reason": float(out["deletion_stats_reason"]["selected_gt_random_rate"].detach().cpu()),
                "deletion_selected_gt_random_rate": float(0.5 * (out["deletion_stats_action"]["selected_gt_random_rate"].detach() + out["deletion_stats_reason"]["selected_gt_random_rate"].detach()).cpu()),
                "pu_stats": out["pu_state"]["stats"],
                "theta_delta_action_abs_mean": float(out["theta_delta_action"].detach().abs().mean().cpu()),
                "theta_delta_reason_abs_mean": float(out["theta_delta_reason"].detach().abs().mean().cpu()),
                "credit_rows": build_target_credit_rows(epoch, out, action, reason),
            }
            valid_pairs_action = int(out["deletion_stats_action"].get("stats", {}).get("valid_pairs", 0))
            valid_pairs_reason = int(out["deletion_stats_reason"].get("stats", {}).get("valid_pairs", 0))
            valid_pairs = valid_pairs_action + valid_pairs_reason
            if valid_pairs > 0:
                epoch_deletion_summary = {
                    "deletion_gap_mean": last_train_stats["deletion_gap_mean"],
                    "deletion_gap_mean_action": last_train_stats["deletion_gap_mean_action"],
                    "deletion_gap_mean_reason": last_train_stats["deletion_gap_mean_reason"],
                    "deletion_selected_gt_random_rate": last_train_stats["deletion_selected_gt_random_rate"],
                    "deletion_selected_gt_random_rate_action": last_train_stats["deletion_selected_gt_random_rate_action"],
                    "deletion_selected_gt_random_rate_reason": last_train_stats["deletion_selected_gt_random_rate_reason"],
                    "valid_pairs": valid_pairs,
                    "valid_pairs_action": valid_pairs_action,
                    "valid_pairs_reason": valid_pairs_reason,
                }
                epoch_credit_rows = last_train_stats["credit_rows"]
            if step % 200 == 0:
                print("tfc_batch " + json.dumps(row), flush=True)
                append_jsonl(out_dir / "loss_components.jsonl", row)
            if args.max_train_samples and step * batch_size >= args.max_train_samples:
                break
        set_trainable(model, base_requires, "calalign")
        threshold_optimizer.zero_grad(set_to_none=True)
        calib_loss_value = 0.0
        calib_steps = 0
        for calib_step, batch in enumerate(train_calib_loader):
            img = batch["image"].to(device, non_blocking=True)
            action = batch["action"].to(device)
            reason = batch["reason"].to(device)
            out = model(img, action, reason, epoch=epoch, split="train", run_deletion=False)
            t_loss = calalign_softf1_loss(out["action_logits_deploy"], out["reason_logits_deploy"], action, reason, out["pu_state"])
            t_loss = t_loss + threshold_smooth_loss(out["theta_delta_action"], out["theta_delta_reason"])
            if not torch.isfinite(t_loss):
                write_json(out_dir / "run_stop_reason.json", {"reason": "nan_or_inf_threshold_loss", "epoch": epoch, "step": calib_step})
                raise RuntimeError("NaN/Inf TFC threshold loss")
            t_loss.backward()
            threshold_lrs = apply_lr_schedule(threshold_optimizer, epoch + 1.0, epochs, train_cfg)
            threshold_optimizer.step()
            threshold_optimizer.zero_grad(set_to_none=True)
            calib_loss_value += float(t_loss.detach().cpu())
            calib_steps += 1
            if args.max_train_samples and calib_step >= 1:
                break
        set_trainable(model, base_requires, "all")
        metrics = evaluate(model, test_loader, device, epoch, out_dir)
        row = {"epoch": epoch, **metrics}
        append_jsonl(out_dir / "metrics_summary.jsonl", row)
        epoch_dir = out_dir / f"epoch_{epoch:03d}"
        deletion_summary = epoch_deletion_summary or {
            "deletion_gap_mean": 0.0,
            "deletion_gap_mean_action": 0.0,
            "deletion_gap_mean_reason": 0.0,
            "deletion_selected_gt_random_rate": 0.0,
            "deletion_selected_gt_random_rate_action": 0.0,
            "deletion_selected_gt_random_rate_reason": 0.0,
            "valid_pairs": 0,
            "valid_pairs_action": 0,
            "valid_pairs_reason": 0,
        }
        write_json(epoch_dir / "run_manifest.json", json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8")))
        append_jsonl(epoch_dir / "loss_components.jsonl", last_loss_row)
        append_jsonl(out_dir / "threshold_train_calib_loss.jsonl", {"epoch": epoch, "loss": calib_loss_value / max(calib_steps, 1), "steps": calib_steps, "lr_threshold": threshold_lrs.get("threshold", threshold_optimizer.param_groups[0]["lr"]) if calib_steps else threshold_optimizer.param_groups[0]["lr"]})
        factor_row = {"epoch": epoch, **{k: v for k, v in last_train_stats.items() if k.startswith("factor_")}}
        append_jsonl(out_dir / "factor_measurement_stats.jsonl", factor_row)
        append_jsonl(epoch_dir / "factor_measurement_stats.jsonl", factor_row)
        credit_rows_to_write = epoch_credit_rows or last_train_stats.get("credit_rows", [])
        if not epoch_credit_rows:
            credit_rows_to_write = [
                {
                    **credit_row,
                    "deletion_selected": 0.0,
                    "deletion_random": 0.0,
                    "selected_vs_random_gap": 0.0,
                    "deletion_available": False,
                }
                for credit_row in credit_rows_to_write
            ]
        for credit_row in credit_rows_to_write:
            append_jsonl(out_dir / "target_credit_stats.jsonl", credit_row)
            append_jsonl(epoch_dir / "target_credit_stats.jsonl", credit_row)
        del_row = {
            "epoch": epoch,
            "selected_vs_random_gap_mean": deletion_summary["deletion_gap_mean"],
            "selected_vs_random_gap_mean_action": deletion_summary["deletion_gap_mean_action"],
            "selected_vs_random_gap_mean_reason": deletion_summary["deletion_gap_mean_reason"],
            "selected_gt_random_rate": deletion_summary["deletion_selected_gt_random_rate"],
            "selected_gt_random_rate_action": deletion_summary["deletion_selected_gt_random_rate_action"],
            "selected_gt_random_rate_reason": deletion_summary["deletion_selected_gt_random_rate_reason"],
            "valid_pairs": deletion_summary["valid_pairs"],
            "valid_pairs_action": deletion_summary["valid_pairs_action"],
            "valid_pairs_reason": deletion_summary["valid_pairs_reason"],
        }
        append_jsonl(out_dir / "deletion_contrast_stats.jsonl", del_row)
        append_jsonl(epoch_dir / "deletion_contrast_stats.jsonl", del_row)
        pu_row = {"epoch": epoch, **last_train_stats.get("pu_stats", {})}
        append_jsonl(out_dir / "pu_state_stats.jsonl", pu_row)
        append_jsonl(epoch_dir / "pu_state_stats.jsonl", pu_row)
        th_row = {
            "epoch": epoch,
            "train_calib_theta": True,
            "theta_delta_action_abs_mean": last_train_stats.get("theta_delta_action_abs_mean", 0.0),
            "theta_delta_reason_abs_mean": last_train_stats.get("theta_delta_reason_abs_mean", 0.0),
            "deploy_oracle_gap_action": metrics.get("Act_oracle_mF1", metrics["Act_mF1"]) - metrics["Act_mF1"],
            "deploy_oracle_gap_reason": metrics.get("Exp_oracle_mF1", metrics["Exp_mF1"]) - metrics["Exp_mF1"],
            "threshold_input_stopgrad_check": True,
            "threshold_optimizer_source": "train_calib_only",
        }
        append_jsonl(out_dir / "threshold_stats.jsonl", th_row)
        append_jsonl(epoch_dir / "threshold_stats.jsonl", th_row)
        pareto_row = firewall_gradient_probe(model, probe_batch, device, epoch) if probe_batch is not None else {
            "epoch": epoch,
            "enabled": "structural_firewall",
            "cosine_action_reason": None,
            "cosine_action_factor": None,
            "cosine_action_credit": None,
            "projection_count": 0,
            "firewall_pass": False,
            "missing_probe_batch": True,
        }
        append_jsonl(out_dir / "pareto_gradient_stats.jsonl", pareto_row)
        append_jsonl(epoch_dir / "pareto_gradient_stats.jsonl", pareto_row)
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics}
        torch.save(ckpt, out_dir / "checkpoint_latest.pth")
        if metrics["joint"] > best_joint:
            best_joint = metrics["joint"]
            torch.save(ckpt, out_dir / "checkpoint_best_test_joint.pth")
        if metrics["Act_mF1"] > best_action_mf1:
            best_action_mf1 = metrics["Act_mF1"]
            torch.save(ckpt, out_dir / "checkpoint_best_test_action_mf1.pth")
        if metrics["Exp_mF1"] > best_exp_mf1:
            best_exp_mf1 = metrics["Exp_mF1"]
            torch.save(ckpt, out_dir / "checkpoint_best_test_exp_mf1.pth")
        collapse_values = [
            last_train_stats.get("factor_action_prob_mean", 0.5),
            last_train_stats.get("factor_reason_prob_mean", 0.5),
            last_train_stats.get("factor_action_rho_mean", 0.5),
            last_train_stats.get("factor_reason_rho_mean", 0.5),
        ]
        collapse = any(v < 0.05 or v > 0.95 for v in collapse_values)
        qrho_collapse_epochs = qrho_collapse_epochs + 1 if collapse else 0
        stop_reason = None
        if qrho_collapse_epochs >= 2:
            stop_reason = {"reason": "q_rho_collapse_for_2_epochs", "epoch": epoch, "values": collapse_values}
        elif epoch > 5 and deletion_summary["deletion_gap_mean"] <= 0.0:
            stop_reason = {"reason": "selected_vs_random_deletion_gap_non_positive_after_epoch5", "epoch": epoch, "gap": deletion_summary["deletion_gap_mean"], "valid_pairs": deletion_summary["valid_pairs"]}
        elif float(last_train_stats.get("pu_stats", {}).get("hard_negative_rate", 0.0)) > float(cfg.get("pu", {}).get("max_hard_negative_rate", 0.20)):
            stop_reason = {"reason": "hard_negative_rate_exceeds_configured_max", "epoch": epoch, "hard_negative_rate": last_train_stats.get("pu_stats", {}).get("hard_negative_rate", 0.0)}
        branch = json.loads((epoch_dir / "action_branch_metrics.json").read_text(encoding="utf-8"))
        if stop_reason is None and epoch >= 6 and any(t > f for t, f in zip(branch.get("TP_to_FN", []), branch.get("FP_to_TP", []))):
            stop_reason = {"reason": "tp_to_fn_exceeds_fp_to_tp_after_action_delta_start", "epoch": epoch, "TP_to_FN": branch.get("TP_to_FN", []), "FP_to_TP": branch.get("FP_to_TP", [])}
        oracle_act = metrics.get("Act_oracle_mF1")
        if epoch >= 6 and prev_oracle_act_mf1 is not None and oracle_act is not None and (prev_oracle_act_mf1 - oracle_act) > 0.004:
            oracle_act_drop_epochs += 1
        else:
            oracle_act_drop_epochs = 0
        if stop_reason is None and epoch >= 6 and oracle_act_drop_epochs >= 2:
            stop_reason = {
                "reason": "oracle_act_mf1_drops_for_2_epochs_after_action_delta_start",
                "epoch": epoch,
                "prev_oracle_act_mf1": prev_oracle_act_mf1,
                "current_oracle_act_mf1": oracle_act,
                "consecutive_drop_epochs": oracle_act_drop_epochs,
            }
        current_act_map = metrics.get("Act_mAP")
        current_exp_mf1 = metrics.get("Exp_mF1")
        current_exp_gap = metrics.get("Exp_oracle_mF1", current_exp_mf1) - current_exp_mf1 if current_exp_mf1 is not None else None
        if (
            stop_reason is None
            and epoch >= 6
            and prev_act_map is not None
            and prev_exp_mf1 is not None
            and prev_exp_deploy_oracle_gap is not None
            and current_act_map is not None
            and current_exp_mf1 is not None
            and current_exp_gap is not None
            and current_act_map < prev_act_map
            and current_exp_mf1 > prev_exp_mf1
            and current_exp_gap >= prev_exp_deploy_oracle_gap
        ):
            stop_reason = {
                "reason": "act_map_drops_while_exp_rises_through_threshold_movement",
                "epoch": epoch,
                "prev_act_map": prev_act_map,
                "current_act_map": current_act_map,
                "prev_exp_mf1": prev_exp_mf1,
                "current_exp_mf1": current_exp_mf1,
                "prev_exp_deploy_oracle_gap": prev_exp_deploy_oracle_gap,
                "current_exp_deploy_oracle_gap": current_exp_gap,
            }
        prev_oracle_act_mf1 = oracle_act
        prev_act_map = current_act_map
        prev_exp_mf1 = current_exp_mf1
        prev_exp_deploy_oracle_gap = current_exp_gap
        if stop_reason is not None:
            write_json(out_dir / "run_stop_reason.json", stop_reason)
            print("tfc_stop " + json.dumps(stop_reason), flush=True)
            break
        print(f"tfc_epoch epoch={epoch} Act_mF1={metrics['Act_mF1']:.6f} Act_oF1={metrics['Act_oF1']:.6f} Exp_mF1={metrics['Exp_mF1']:.6f} Exp_oF1={metrics['Exp_oF1']:.6f} joint={metrics['joint']:.6f}", flush=True)


if __name__ == "__main__":
    main()
