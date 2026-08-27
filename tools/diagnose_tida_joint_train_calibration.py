import argparse
import json
from pathlib import Path

import torch

from fate_oia.utils.tida_object_intent_metrics import (
    combine_object_intent_utility_policies,
    fit_object_intent_utility_policy_oof,
)


def load(path):
    return torch.load(path, map_location="cpu")


def per_label_f1(logits, target, thresholds):
    pred = logits.sigmoid() >= thresholds[None]
    pos = target > 0.5
    tp = (pred & pos).sum(0).float()
    fp = (pred & ~pos).sum(0).float()
    fn = (~pred & pos).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--locked-thresholds", default="0.65,0.68,0.60,0.60")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.policy_rows)
    locked = torch.tensor([float(v) for v in args.locked_thresholds.split(",")])
    base = rows["pre_object_intent_action"].float()
    candidate = rows["object_intent_action_candidate"].float()
    target = rows["action_target"].float()
    common = dict(
        base_logits=base,
        candidate_delta=candidate,
        target=target,
        locked_thresholds=locked,
        scales=(0.0, 4.0, 8.0, 16.0, 24.0, 32.0, 48.0, 64.0),
        cutoffs=(0.0, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8),
        folds=5,
        max_selected_rate=0.5,
        min_selected_benefit_rate=0.65,
        min_positive_fold_fraction=0.4,
        min_non_degrading_fold_fraction=0.8,
        fold_degradation_tolerance=0.002,
        cap=0.08,
    )
    directional = fit_object_intent_utility_policy_oof(
        utility_gate=rows["object_intent_action_directional_utility_gate"].float(), **common
    )
    risk = fit_object_intent_utility_policy_oof(
        utility_gate=rows["object_intent_action_risk_utility_gate"].float(), **common
    )
    policy = combine_object_intent_utility_policies(directional, risk)
    source = policy["utility_source"]
    train_gate = torch.where(
        source[None] == 1,
        rows["object_intent_action_risk_utility_gate"].float(),
        rows["object_intent_action_directional_utility_gate"].float(),
    )
    train_selected = train_gate >= policy["cutoff"][None]
    train_delta = (policy["scale"][None] * candidate).clamp(-0.08, 0.08)
    train_logits = base + policy["gate"][None] * train_selected * train_delta
    grid = torch.arange(0.30, 0.901, 0.005)
    thresholds = []
    train_f1 = []
    for label in range(4):
        scores = torch.stack([
            per_label_f1(train_logits[:, label:label+1], target[:, label:label+1], value[None])[0]
            for value in grid
        ])
        best = int(scores.argmax())
        thresholds.append(grid[best])
        train_f1.append(scores[best])
    thresholds = torch.stack(thresholds)
    test_base = load(args.epoch_dir / "pre_object_intent_action_test.pt").float()
    test_candidate = load(args.epoch_dir / "object_intent_action_candidate_test.pt").float()
    test_target = load(args.epoch_dir / "action_target_test.pt").float()
    test_directional = load(args.epoch_dir / "object_intent_action_directional_utility_gate_test.pt").float()
    test_risk = load(args.epoch_dir / "object_intent_action_risk_utility_gate_test.pt").float()
    test_gate = torch.where(source[None] == 1, test_risk, test_directional)
    test_selected = test_gate >= policy["cutoff"][None]
    test_logits = test_base + policy["gate"][None] * test_selected * (
        policy["scale"][None] * test_candidate
    ).clamp(-0.08, 0.08)
    locked_f1 = per_label_f1(test_logits, test_target, locked)
    refit_f1 = per_label_f1(test_logits, test_target, thresholds)
    branch_results = {}
    for name, branch_policy, branch_gate in (
        ("directional_only", directional, test_directional),
        ("risk_only", risk, test_risk),
    ):
        branch_selected = branch_gate >= branch_policy["cutoff"][None]
        branch_logits = test_base + branch_policy["gate"][None] * branch_selected * (
            branch_policy["scale"][None] * test_candidate
        ).clamp(-0.08, 0.08)
        branch_f1 = per_label_f1(branch_logits, test_target, locked)
        branch_results[name] = {
            "test_mf1": float(branch_f1.mean()),
            "test_per_action_f1": branch_f1.tolist(),
            "scale": branch_policy["scale"].tolist(),
            "cutoff": branch_policy["cutoff"].tolist(),
            "oof_gain": branch_policy["oof_gain"].tolist(),
        }
    result = {
        "protocol": "policy and thresholds fit on 8031 train-only rows; test read once",
        "thresholds": thresholds.tolist(),
        "locked_thresholds": locked.tolist(),
        "train_per_action_f1": torch.stack(train_f1).tolist(),
        "test_locked_per_action_f1": locked_f1.tolist(),
        "test_locked_mf1": float(locked_f1.mean()),
        "test_refit_per_action_f1": refit_f1.tolist(),
        "test_refit_mf1": float(refit_f1.mean()),
        "test_gain_over_locked": float(refit_f1.mean() - locked_f1.mean()),
        "policy_scale": policy["scale"].tolist(),
        "policy_cutoff": policy["cutoff"].tolist(),
        "policy_source": source.tolist(),
        "utility_branch_diagnostic": branch_results,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
