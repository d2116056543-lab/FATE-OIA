from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")


def per_label_f1(logits: torch.Tensor, labels: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    prediction = torch.sigmoid(logits) >= thresholds
    truth = labels > 0.5
    tp = (prediction & truth).sum(0).float()
    fp = (prediction & ~truth).sum(0).float()
    fn = (~prediction & truth).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def fit_thresholds(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.02, 0.98, 97)
    result = []
    for label_index in range(logits.shape[1]):
        prediction = torch.sigmoid(logits[:, label_index])[:, None] >= grid[None]
        truth = labels[:, label_index, None] > 0.5
        tp = (prediction & truth).sum(0).float()
        fp = (prediction & ~truth).sum(0).float()
        fn = (~prediction & truth).sum(0).float()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        result.append(grid[f1.argmax()])
    return torch.stack(result)


def average_precision(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for label_index in range(logits.shape[1]):
        truth = labels[logits[:, label_index].argsort(descending=True), label_index]
        precision = truth.cumsum(0) / torch.arange(1, len(truth) + 1)
        result.append((precision * truth).sum() / truth.sum().clamp_min(1))
    return torch.stack(result)


def roc_auc(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    result = []
    for label_index in range(logits.shape[1]):
        scores = logits[:, label_index]
        truth = labels[:, label_index] > 0.5
        positives = truth.sum()
        negatives = (~truth).sum()
        if positives == 0 or negatives == 0:
            result.append(torch.tensor(float("nan")))
            continue
        order = scores.argsort()
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
        auc = (ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        result.append(auc)
    return torch.stack(result)


def cv_threshold_f1(
    logits: torch.Tensor,
    labels: torch.Tensor,
    folds: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.zeros_like(logits, dtype=torch.bool)
    for held_out in range(thresholds.shape[0]):
        mask = folds == held_out
        prediction[mask] = torch.sigmoid(logits[mask]) >= thresholds[held_out]
    truth = labels > 0.5
    tp = (prediction & truth).sum(0).float()
    fp = (prediction & ~truth).sum(0).float()
    fn = (~prediction & truth).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def build_future_intent_features(motion: torch.Tensor) -> torch.Tensor:
    x, y = motion[..., 0], motion[..., 1]
    vx, vy = motion[..., 2], motion[..., 3]
    visibility = motion[..., 11].clamp(0, 1)
    confidence = motion[..., 12].clamp(0, 1)
    support = (visibility * confidence.sqrt()).clamp_min(1e-4)
    centres = torch.tensor([0.0, 0.0, -0.55, 0.55])
    widths = torch.tensor([0.28, 0.75, 0.30, 0.30])
    result = []
    for action_index in range(4):
        action_features = []
        for horizon in (0.0, 2.0, 4.0, 8.0):
            future_x = x + horizon * vx
            future_y = y + horizon * vy
            corridor = torch.exp(-((future_x - centres[action_index]) / widths[action_index]).square())
            front = torch.sigmoid(3.0 * future_y)
            weight = support * corridor * front
            normalizer = weight.sum(-1, keepdim=True).clamp_min(1e-5)
            pooled = torch.einsum("nk,nkf->nf", weight / normalizer, motion)
            action_features.extend(
                (pooled, weight.sum(-1, keepdim=True), weight.amax(-1, keepdim=True))
            )
        result.append(torch.cat(action_features, dim=-1))
    return torch.stack(result, dim=1)


def evaluate_task(
    name: str,
    base: torch.Tensor,
    labels: torch.Tensor,
    fixed_thresholds: torch.Tensor,
    features: torch.Tensor,
    folds: torch.Tensor,
    max_delta: float,
) -> dict[str, object]:
    num_labels = base.shape[1]
    if features.ndim == 3 and features.shape[1] == num_labels:
        feature_mode = "per_label"
        input_dim = features.shape[-1]
    else:
        feature_mode = "shared"
        features = features.flatten(1)
        input_dim = features.shape[-1]

    corrected = torch.zeros_like(base)
    corrected_thresholds = torch.zeros(5, num_labels)
    base_thresholds = torch.zeros(5, num_labels)
    for held_out in range(5):
        train = folds != held_out
        test = ~train
        if feature_mode == "per_label":
            mean = features[train].mean((0, 1), keepdim=True)
            std = features[train].std((0, 1), keepdim=True).clamp_min(1e-4)
            model = nn.Linear(input_dim, 1)
        else:
            mean = features[train].mean(0, keepdim=True)
            std = features[train].std(0, keepdim=True).clamp_min(1e-4)
            model = nn.Linear(input_dim, num_labels)
        nn.init.zeros_(model.weight)
        nn.init.zeros_(model.bias)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.10)
        positive = labels[train].sum(0)
        negative = train.sum() - positive
        pos_weight = (negative / positive.clamp_min(1)).clamp(1.0, 8.0)
        for _ in range(500):
            normalized = (features[train] - mean) / std
            delta = max_delta * torch.tanh(model(normalized).squeeze(-1))
            prediction = base[train] + delta
            loss = F.binary_cross_entropy_with_logits(
                prediction, labels[train], pos_weight=pos_weight
            ) + 0.05 * delta.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_delta = max_delta * torch.tanh(model((features[train] - mean) / std).squeeze(-1))
            test_delta = max_delta * torch.tanh(model((features[test] - mean) / std).squeeze(-1))
            corrected[test] = base[test] + test_delta
            corrected_thresholds[held_out] = fit_thresholds(base[train] + train_delta, labels[train])
            base_thresholds[held_out] = fit_thresholds(base[train], labels[train])

    base_ap = average_precision(base, labels)
    corrected_ap = average_precision(corrected, labels)
    base_auc = roc_auc(base, labels)
    corrected_auc = roc_auc(corrected, labels)
    base_fixed_f1 = per_label_f1(base, labels, fixed_thresholds)
    corrected_fixed_f1 = per_label_f1(corrected, labels, fixed_thresholds)
    base_cv_f1 = cv_threshold_f1(base, labels, folds, base_thresholds)
    corrected_cv_f1 = cv_threshold_f1(corrected, labels, folds, corrected_thresholds)
    return {
        "task": name,
        "samples": len(labels),
        "base_fixed_mf1": float(base_fixed_f1.mean()),
        "corrected_fixed_mf1": float(corrected_fixed_f1.mean()),
        "base_cv_threshold_mf1": float(base_cv_f1.mean()),
        "corrected_cv_threshold_mf1": float(corrected_cv_f1.mean()),
        "base_map": float(base_ap.mean()),
        "corrected_map": float(corrected_ap.mean()),
        "map_delta": float(corrected_ap.mean() - base_ap.mean()),
        "base_macro_auc": float(torch.nanmean(base_auc)),
        "corrected_macro_auc": float(torch.nanmean(corrected_auc)),
        "auc_delta": float(torch.nanmean(corrected_auc - base_auc)),
        "base_bce": float(F.binary_cross_entropy_with_logits(base, labels)),
        "corrected_bce": float(F.binary_cross_entropy_with_logits(corrected, labels)),
        "per_label_ap_delta": (corrected_ap - base_ap).tolist(),
        "per_label_auc_delta": (corrected_auc - base_auc).tolist(),
        "per_label_fixed_f1_delta": (corrected_fixed_f1 - base_fixed_f1).tolist(),
        "per_label_cv_f1_delta": (corrected_cv_f1 - base_cv_f1).tolist(),
    }


def main() -> None:
    motion = torch.load(ROOT / "cotracker_motion_features_test.pt", map_location="cpu").float()
    future_features = build_future_intent_features(motion)
    calibration = torch.tensor(json.loads((ROOT / "calibration.json").read_text())["image"])
    action_base = torch.load(ROOT / "pre_relational_action_test.pt", map_location="cpu").float()
    action_target = torch.load(ROOT / "action_target_test.pt", map_location="cpu").float()
    reason_base = torch.load(ROOT / "pre_relational_reason_test.pt", map_location="cpu").float()
    reason_target = torch.load(ROOT / "reason_target_test.pt", map_location="cpu").float()
    folds = torch.randperm(len(action_target), generator=torch.Generator().manual_seed(20260825)) % 5
    action_result = evaluate_task(
        "action", action_base, action_target, calibration[:4], future_features, folds, 0.25
    )
    reason_result = evaluate_task(
        "reason", reason_base, reason_target, calibration[4:], future_features, folds, 0.20
    )
    output = {"motion_source": "CoTracker3_offline_grid8", "action": action_result, "reason": reason_result}
    output_path = ROOT / "cotracker_future_intent_cv.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
