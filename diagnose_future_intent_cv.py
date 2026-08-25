from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(r"F:\FATE_Drive_runs\tida_relational_v8_2_pilot5584x1_retry2\epoch_000")
motion = torch.load(ROOT / "relational_motion_features_test.pt", map_location="cpu").float()
base = torch.load(ROOT / "pre_relational_action_test.pt", map_location="cpu").float()
target = torch.load(ROOT / "action_target_test.pt", map_location="cpu").float()
threshold = torch.tensor(json.loads((ROOT / "calibration.json").read_text())["image"][:4])

x, y = motion[..., 0], motion[..., 1]
vx, vy = motion[..., 2], motion[..., 3]
visibility = motion[..., 11].clamp(0, 1)
anchor = motion[..., 12].clamp(0, 1)
support = (visibility * anchor.sqrt()).clamp_min(1e-4)
centres = torch.tensor([0.0, 0.0, -0.55, 0.55])
widths = torch.tensor([0.28, 0.75, 0.30, 0.30])
features = []
for action in range(4):
    action_features = []
    for horizon in (0.0, 2.0, 4.0, 8.0):
        future_x = x + horizon * vx
        future_y = y + horizon * vy
        corridor = torch.exp(-((future_x - centres[action]) / widths[action]).square())
        front = torch.sigmoid(3.0 * future_y)
        weight = support * corridor * front
        norm = weight.sum(-1, keepdim=True).clamp_min(1e-5)
        pooled = torch.einsum("nk,nkf->nf", weight / norm, motion)
        action_features.extend((pooled, weight.sum(-1, keepdim=True), weight.amax(-1, keepdim=True)))
    features.append(torch.cat(action_features, dim=-1))
features = torch.stack(features, dim=1)


def per_label_f1(logits, labels, thresholds):
    pred = torch.sigmoid(logits) >= thresholds
    truth = labels > 0.5
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def fit_thresholds(logits, labels):
    grid = torch.linspace(0.05, 0.95, 91)
    chosen = []
    for label in range(4):
        pred = torch.sigmoid(logits[:, label])[:, None] >= grid[None]
        truth = labels[:, label, None] > 0.5
        tp = (pred & truth).sum(0).float(); fp = (pred & ~truth).sum(0).float(); fn = (~pred & truth).sum(0).float()
        chosen.append(grid[(2 * tp / (2 * tp + fp + fn).clamp_min(1)).argmax()])
    return torch.stack(chosen)


def average_precision(logits, labels):
    result = []
    for label in range(4):
        truth = labels[logits[:, label].argsort(descending=True), label]
        result.append((truth.cumsum(0) / torch.arange(1, len(truth) + 1) * truth).sum() / truth.sum().clamp_min(1))
    return torch.stack(result).mean()


fold = torch.randperm(len(target), generator=torch.Generator().manual_seed(20260825)) % 5
prediction = torch.zeros_like(base)
cv_thresholds = torch.zeros(5, 4)
for held_out in range(5):
    train, test = fold != held_out, fold == held_out
    mean = features[train].mean((0, 1), keepdim=True)
    std = features[train].std((0, 1), keepdim=True).clamp_min(1e-4)
    model = nn.Linear(features.shape[-1], 1)
    nn.init.zeros_(model.weight); nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.1)
    for _ in range(400):
        delta = 0.20 * torch.tanh(model((features[train] - mean) / std).squeeze(-1))
        loss = F.binary_cross_entropy_with_logits(base[train] + delta, target[train]) + 0.05 * delta.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        train_logits = base[train] + 0.20 * torch.tanh(model((features[train] - mean) / std).squeeze(-1))
        prediction[test] = base[test] + 0.20 * torch.tanh(model((features[test] - mean) / std).squeeze(-1))
        cv_thresholds[held_out] = fit_thresholds(train_logits, target[train])

pred = torch.zeros_like(prediction, dtype=torch.bool)
for held_out in range(5):
    test = fold == held_out
    pred[test] = torch.sigmoid(prediction[test]) >= cv_thresholds[held_out]
truth = target > 0.5
tp = (pred & truth).sum(0).float(); fp = (pred & ~truth).sum(0).float(); fn = (~pred & truth).sum(0).float()
print(json.dumps({
    "base_fixed_mf1": float(per_label_f1(base, target, threshold).mean()),
    "future_intent_fixed_mf1": float(per_label_f1(prediction, target, threshold).mean()),
    "future_intent_cv_threshold_mf1": float((2 * tp / (2 * tp + fp + fn).clamp_min(1)).mean()),
    "base_map": float(average_precision(base, target)),
    "future_intent_map": float(average_precision(prediction, target)),
    "base_bce": float(F.binary_cross_entropy_with_logits(base, target)),
    "future_intent_bce": float(F.binary_cross_entropy_with_logits(prediction, target)),
}, indent=2))
