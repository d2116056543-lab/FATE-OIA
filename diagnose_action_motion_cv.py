from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")
motion = torch.load(ROOT / "relational_motion_features_test.pt", map_location="cpu").float()
attention = torch.load(ROOT / "relational_action_attention_test.pt", map_location="cpu").float()
base = torch.load(ROOT / "pre_relational_action_test.pt", map_location="cpu").float()
target = torch.load(ROOT / "action_target_test.pt", map_location="cpu").float()
threshold = torch.tensor(json.loads((ROOT / "calibration.json").read_text())["image"][:4])
features = torch.einsum("nlk,nkf->nlf", attention, motion)
features = torch.cat((features, features.square(), features.abs()), dim=-1)


def mf1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = torch.sigmoid(logits) >= threshold
    truth = labels > 0.5
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())


def fit_thresholds(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    grid = torch.linspace(0.05, 0.95, 91)
    chosen = []
    for label in range(logits.shape[1]):
        scores = torch.sigmoid(logits[:, label])[:, None] >= grid[None]
        truth = labels[:, label, None] > 0.5
        tp = (scores & truth).sum(0).float()
        fp = (scores & ~truth).sum(0).float()
        fn = (~scores & truth).sum(0).float()
        chosen.append(grid[(2 * tp / (2 * tp + fp + fn).clamp_min(1)).argmax()])
    return torch.stack(chosen)


def average_precision(logits: torch.Tensor, labels: torch.Tensor) -> float:
    values = []
    for label in range(logits.shape[1]):
        order = logits[:, label].argsort(descending=True)
        truth = labels[order, label]
        positives = truth.sum().clamp_min(1)
        precision = truth.cumsum(0) / torch.arange(1, len(truth) + 1)
        values.append((precision * truth).sum() / positives)
    return float(torch.stack(values).mean())


generator = torch.Generator().manual_seed(20260825)
fold = torch.randperm(len(target), generator=generator) % 5
prediction = torch.zeros_like(base)
base_cv_prediction = torch.zeros_like(base)
prediction_thresholds = torch.zeros(5, 4)
base_thresholds = torch.zeros(5, 4)
scale_values = []
for held_out in range(5):
    train = fold != held_out
    test = ~train
    mean = features[train].mean((0, 1), keepdim=True)
    std = features[train].std((0, 1), keepdim=True).clamp_min(1e-4)
    model = nn.Linear(features.shape[-1], 1)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.05)
    for _ in range(300):
        delta = 0.25 * torch.tanh(model((features[train] - mean) / std).squeeze(-1))
        loss = F.binary_cross_entropy_with_logits(base[train] + delta, target[train])
        loss = loss + 0.02 * delta.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_prediction = base[train] + 0.25 * torch.tanh(
            model((features[train] - mean) / std).squeeze(-1)
        )
        prediction[test] = base[test] + 0.25 * torch.tanh(
            model((features[test] - mean) / std).squeeze(-1)
        )
        base_cv_prediction[test] = base[test]
        prediction_thresholds[held_out] = fit_thresholds(train_prediction, target[train])
        base_thresholds[held_out] = fit_thresholds(base[train], target[train])
        scale_values.append(float(model.weight.norm()))


def cv_mf1(logits: torch.Tensor, thresholds_by_fold: torch.Tensor) -> float:
    pred = torch.zeros_like(logits, dtype=torch.bool)
    for held_out in range(5):
        test = fold == held_out
        pred[test] = torch.sigmoid(logits[test]) >= thresholds_by_fold[held_out]
    truth = target > 0.5
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean())

print(json.dumps({
    "samples": len(target),
    "base_mf1": mf1(base, target),
    "five_fold_motion_corrected_mf1": mf1(prediction, target),
    "five_fold_refit_base_mf1": cv_mf1(base_cv_prediction, base_thresholds),
    "five_fold_refit_motion_corrected_mf1": cv_mf1(prediction, prediction_thresholds),
    "base_map": average_precision(base, target),
    "five_fold_motion_corrected_map": average_precision(prediction, target),
    "base_bce": float(F.binary_cross_entropy_with_logits(base, target)),
    "five_fold_motion_corrected_bce": float(F.binary_cross_entropy_with_logits(prediction, target)),
    "fold_weight_norms": scale_values,
}, indent=2))
