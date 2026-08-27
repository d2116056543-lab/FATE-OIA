import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from fate_oia.utils.tida_object_intent_metrics import fit_object_intent_utility_policy_oof


def _load(path: Path):
    return torch.load(path, map_location="cpu")


def _features(base, candidate, directional, risk, thresholds):
    threshold_logits = torch.logit(thresholds.clamp(1e-4, 1 - 1e-4))[None]
    margin = base - threshold_logits
    eps = 1e-5
    directional_logit = torch.logit(directional.clamp(eps, 1 - eps))
    risk_logit = torch.logit(risk.clamp(eps, 1 - eps))
    return torch.stack(
        [
            directional_logit,
            risk_logit,
            margin,
            margin.abs(),
            candidate,
            candidate.abs(),
            directional_logit * risk_logit,
            candidate.abs() / (margin.abs() + 0.05),
        ],
        dim=-1,
    )


def _macro_f1(logits, target, thresholds):
    pred = logits.sigmoid() >= thresholds[None]
    positive = target > 0.5
    tp = (pred & positive).sum(0).float()
    fp = (pred & ~positive).sum(0).float()
    fn = (~pred & positive).sum(0).float()
    per_label = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
    return float(per_label.mean()), per_label.tolist()


def _fit_linear(train_x, train_y, eval_x, steps=300):
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-4)
    x = (train_x - mean) / std
    z = (eval_x - mean) / std
    weight = torch.zeros(x.shape[1], requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.03, weight_decay=1e-3)
    for _ in range(steps):
        logits = x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return torch.sigmoid(z @ weight.detach() + bias.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(args.policy_rows)
    epoch = args.epoch_dir
    thresholds = torch.tensor([float(v) for v in args.thresholds.split(",")])
    train_base = rows["pre_object_intent_action"].float()
    train_candidate = rows["object_intent_action_candidate"].float()
    train_directional = rows["object_intent_action_directional_utility_gate"].float()
    train_risk = rows["object_intent_action_risk_utility_gate"].float()
    train_target = rows["action_target"].float()
    test_base = _load(epoch / "pre_object_intent_action_test.pt").float()
    test_candidate = _load(epoch / "object_intent_action_candidate_test.pt").float()
    test_directional = _load(epoch / "object_intent_action_directional_utility_gate_test.pt").float()
    test_risk = _load(epoch / "object_intent_action_risk_utility_gate_test.pt").float()
    test_target = _load(epoch / "action_target_test.pt").float()
    train_x = _features(train_base, train_candidate, train_directional, train_risk, thresholds)
    test_x = _features(test_base, test_candidate, test_directional, test_risk, thresholds)
    sign = train_target * 2 - 1
    helpful = (sign * train_candidate > 0).float()
    generator = torch.Generator().manual_seed(3407)
    fold_ids = torch.empty(train_base.shape[0], dtype=torch.long)
    permutation = torch.randperm(train_base.shape[0], generator=generator)
    fold_ids[permutation] = torch.arange(train_base.shape[0]) % 5
    oof_score = torch.zeros_like(train_base)
    test_scores = []
    for fold in range(5):
        fit = fold_ids != fold
        holdout = ~fit
        fold_test = torch.zeros_like(test_base)
        for label in range(4):
            oof_score[holdout, label] = _fit_linear(
                train_x[fit, label], helpful[fit, label], train_x[holdout, label]
            )
            fold_test[:, label] = _fit_linear(
                train_x[fit, label], helpful[fit, label], test_x[:, label]
            )
        test_scores.append(fold_test)
    test_score = torch.stack(test_scores).mean(0)
    policy = fit_object_intent_utility_policy_oof(
        train_base,
        train_candidate,
        oof_score,
        train_target,
        thresholds,
        scales=(0.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 80.0, 96.0),
        cutoffs=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        folds=5,
        max_selected_rate=0.6,
        min_selected_benefit_rate=0.55,
        min_positive_fold_fraction=0.6,
        min_non_degrading_fold_fraction=0.8,
        fold_degradation_tolerance=0.002,
        cap=0.08,
    )
    selected = test_score >= policy["cutoff"][None]
    delta = (policy["scale"][None] * test_candidate).clamp(-0.08, 0.08)
    deployed = test_base + policy["gate"][None] * selected * delta
    base_mf1, base_per = _macro_f1(test_base, test_target, thresholds)
    deploy_mf1, deploy_per = _macro_f1(deployed, test_target, thresholds)
    signed = (test_target * 2 - 1) * delta
    selected_count = selected.sum(0).clamp_min(1)
    result = {
        "protocol": "5-fold train-only crossfit quality; test read after policy lock",
        "base_test_mf1": base_mf1,
        "deploy_test_mf1": deploy_mf1,
        "test_gain": deploy_mf1 - base_mf1,
        "base_per_action_f1": base_per,
        "deploy_per_action_f1": deploy_per,
        "policy_gate": policy["gate"].tolist(),
        "policy_scale": policy["scale"].tolist(),
        "policy_cutoff": policy["cutoff"].tolist(),
        "policy_oof_gain": policy["oof_gain"].tolist(),
        "test_selected_rate": selected.float().mean(0).tolist(),
        "test_selected_benefit_rate": (((signed > 0) & selected).sum(0) / selected_count).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
