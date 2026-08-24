from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from fate_oia.engine.evaluate_tida_oia import branch_metrics, fit_train_calib_thresholds
from fate_oia.engine.train_tida_oia import build_runtime


INPUT_KEYS = (
    "video_action_logits_base", "traffic_trajectory_order_delta",
    "traffic_trajectory_state_features", "traffic_trajectory_support",
    "trajectory_state_strength", "trajectory_interaction_risk",
)


def _macro_f1(logits: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> float:
    prediction = logits.sigmoid() >= threshold
    positive = target > 0.5
    tp = (prediction & positive).sum(0).float()
    fp = (prediction & ~positive).sum(0).float()
    fn = (~prediction & positive).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())


def _fit_thresholds(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.05, 0.95, 91, device=logits.device)
    result = []
    for label in range(target.shape[1]):
        scores = torch.stack([_label_f1(logits[:, label], target[:, label], value) for value in grid])
        result.append(grid[int(scores.argmax())])
    return torch.stack(result)


def _label_f1(logit: torch.Tensor, target: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    prediction = logit.sigmoid() >= threshold
    positive = target > 0.5
    tp = (prediction & positive).sum().float()
    fp = (prediction & ~positive).sum().float()
    fn = (~prediction & positive).sum().float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def _balanced_boundary_loss(
    deploy_logits: torch.Tensor,
    base_logits: torch.Tensor,
    target: torch.Tensor,
    threshold: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    near = torch.exp(-((base_logits.sigmoid() - threshold).abs() / 0.08)).detach()
    weight = 0.25 + 0.75 * near
    value = F.binary_cross_entropy_with_logits(deploy_logits, target, reduction="none")
    terms = []
    for label in range(target.shape[1]):
        for class_value in (0.0, 1.0):
            mask = target[:, label] == class_value
            if mask.any():
                terms.append((value[mask, label] * weight[mask, label]).mean())
    return torch.stack(terms).mean() + 0.02 * delta.square().mean()


@torch.no_grad()
def _collect(model, loader, device: torch.device) -> dict[str, torch.Tensor]:
    rows: dict[str, list[torch.Tensor]] = {key: [] for key in INPUT_KEYS}
    rows.update({"action_target": [], "reason_target": [], "video_reason": [], "image_action": [], "image_reason": []})
    model.eval()
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
        output = model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
        )
        for key in INPUT_KEYS:
            rows[key].append(output[key].detach().cpu())
        rows["action_target"].append(batch["action"].detach().cpu())
        rows["reason_target"].append(batch["reason"].detach().cpu())
        rows["video_reason"].append(output["video_reason_logits"].detach().cpu())
        rows["image_action"].append(output["image_action_logits"].detach().cpu())
        rows["image_reason"].append(output["image_reason_logits"].detach().cpu())
    return {key: torch.cat(value).float() for key, value in rows.items()}


def _head_forward(head, rows: dict[str, torch.Tensor], index: torch.Tensor, device: torch.device):
    values = [rows[key][index].to(device) for key in INPUT_KEYS]
    return head(*values)


def fit_cv(runtime, *, folds: int, steps: int, lr: float) -> tuple[
    list[dict[str, torch.Tensor]], torch.Tensor, dict[str, Any], dict[str, torch.Tensor]
]:
    calib = _collect(runtime.model, runtime.loaders["train_calib"], runtime.device)
    target = calib["action_target"].to(runtime.device)
    count = target.shape[0]
    assignment = torch.arange(count, device=runtime.device).remainder(folds)
    accepted_states = []
    fold_thresholds = []
    diagnostics = []
    for fold in range(folds):
        fit_index = torch.where(assignment != fold)[0].cpu()
        hold_index = torch.where(assignment == fold)[0].cpu()
        head = copy.deepcopy(runtime.model.traffic_adaptive_boundary).to(runtime.device)
        with torch.no_grad():
            head.network[-1].weight.zero_()
            head.network[-1].bias.zero_()
        zero_state = {
            key: value.detach().cpu().clone() for key, value in head.state_dict().items()
        }
        optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
        base_fit = calib["video_action_logits_base"][fit_index].to(runtime.device)
        target_fit = calib["action_target"][fit_index].to(runtime.device)
        base_threshold = _fit_thresholds(base_fit, target_fit)
        base_hold = calib["video_action_logits_base"][hold_index].to(runtime.device)
        target_hold = calib["action_target"][hold_index].to(runtime.device)
        baseline_score = _macro_f1(base_hold, target_hold, base_threshold)
        best_score, best_state, best_threshold, best_step = baseline_score, None, base_threshold, 0
        for step in range(1, steps + 1):
            output = _head_forward(head, calib, fit_index, runtime.device)
            loss = _balanced_boundary_loss(
                output["traffic_adaptive_deploy_action_logits"], base_fit,
                target_fit, base_threshold, output["traffic_adaptive_boundary_delta"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if step % 10 == 0 or step == steps:
                with torch.no_grad():
                    fit_output = _head_forward(head, calib, fit_index, runtime.device)
                    threshold = _fit_thresholds(
                        fit_output["traffic_adaptive_deploy_action_logits"], target_fit
                    )
                    hold_output = _head_forward(head, calib, hold_index, runtime.device)
                    score = _macro_f1(
                        hold_output["traffic_adaptive_deploy_action_logits"], target_hold, threshold
                    )
                if score > best_score:
                    best_score = score
                    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
                    best_threshold = threshold.detach().cpu()
                    best_step = step
        accepted = best_state is not None
        accepted_states.append(best_state or zero_state)
        fold_thresholds.append(best_threshold.cpu())
        diagnostics.append({
            "fold": fold, "fit_count": len(fit_index), "holdout_count": len(hold_index),
            "baseline_holdout_mf1": baseline_score, "best_holdout_mf1": best_score,
            "accepted": accepted, "best_step": best_step,
        })
    return accepted_states, torch.stack(fold_thresholds).median(0).values, {
        "calib_count": count, "folds": diagnostics,
    }, calib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", default="online")
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--context-chunk-size", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()
    runtime = build_runtime(args, evaluation_only=True)
    states, action_threshold, diagnostics, calib_rows = fit_cv(
        runtime, folds=args.folds, steps=args.steps, lr=args.lr
    )
    test = _collect(runtime.model, runtime.loaders["test"], runtime.device)
    deltas = []
    all_index = torch.arange(test["action_target"].shape[0])
    for state in states:
        head = copy.deepcopy(runtime.model.traffic_adaptive_boundary).to(runtime.device)
        head.load_state_dict(state)
        with torch.no_grad():
            deltas.append(_head_forward(head, test, all_index, runtime.device)["traffic_adaptive_boundary_delta"].cpu())
    ensemble_delta = torch.stack(deltas).mean(0)
    deploy_action = test["video_action_logits_base"] - ensemble_delta
    reason_threshold = fit_train_calib_thresholds({
        "video_action": calib_rows["video_action_logits_base"], "video_reason": calib_rows["video_reason"],
        "image_action": calib_rows["image_action"], "image_reason": calib_rows["image_reason"],
        "action_target": calib_rows["action_target"], "reason_target": calib_rows["reason_target"],
    })["video"][4:]
    rows = {
        "image_action": test["image_action"], "video_action": deploy_action,
        "image_reason": test["image_reason"], "video_reason": test["video_reason"],
        "action_target": test["action_target"], "reason_target": test["reason_target"],
    }
    metrics = branch_metrics(rows, torch.cat((action_threshold, reason_threshold)))["video"]
    payload = {
        "test_labels_used_for_fit_or_selection": False,
        "diagnostics": diagnostics,
        "action_thresholds": action_threshold.tolist(),
        "ensemble_delta_rms": float(ensemble_delta.square().mean().sqrt()),
        "metrics": {key: metrics[key] for key in (
            "Act_mF1", "Act_oF1", "Act_mAP", "Act_per_label_f1",
            "Exp_mF1", "Exp_oF1", "Exp_mAP",
        )},
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "traffic_boundary_cv_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save({"head_states": states, "action_thresholds": action_threshold}, output / "traffic_boundary_cv_heads.pt")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
