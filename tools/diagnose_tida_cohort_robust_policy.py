import argparse
import json
from pathlib import Path

import torch


def load(path):
    return torch.load(path, map_location="cpu")


def label_f1(logits, truth, threshold):
    prediction = logits.sigmoid() >= threshold
    positive = truth > 0.5
    tp = (prediction & positive).sum().float()
    fp = (prediction & ~positive).sum().float()
    fn = (~prediction & positive).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-rows", type=Path, required=True)
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load(args.policy_rows)
    sizes = rows["_policy_cohort_sizes"]
    names = list(sizes)
    offsets, start = {}, 0
    for name in names:
        offsets[name] = torch.arange(start, start + int(sizes[name]))
        start += int(sizes[name])
    thresholds = torch.tensor([0.65, 0.68, 0.60, 0.60])
    base = rows["pre_object_intent_action"].float()
    candidate = rows["object_intent_action_candidate"].float()
    target = rows["action_target"].float()
    utilities = [
        rows["object_intent_action_directional_utility_gate"].float(),
        rows["object_intent_action_risk_utility_gate"].float(),
    ]
    test_base = load(args.epoch_dir / "pre_object_intent_action_test.pt").float()
    test_candidate = load(args.epoch_dir / "object_intent_action_candidate_test.pt").float()
    test_target = load(args.epoch_dir / "action_target_test.pt").float()
    test_utilities = [
        load(args.epoch_dir / "object_intent_action_directional_utility_gate_test.pt").float(),
        load(args.epoch_dir / "object_intent_action_risk_utility_gate_test.pt").float(),
    ]
    choices, test_f1 = [], []
    for label in range(4):
        base_group = {
            name: label_f1(base[index, label], target[index, label], thresholds[label])
            for name, index in offsets.items()
        }
        best = {"score": 0.0, "scale": 0.0, "cutoff": 0.0, "source": 0, "gains": {name: 0.0 for name in names}}
        cutoff_values = sorted(set([0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8] + [
            float(utility[:, label].quantile(q)) for utility in utilities for q in (0.5, 0.75)
        ]))
        sign = target[:, label] * 2 - 1
        for source, utility in enumerate(utilities):
            for scale in (4, 8, 16, 24, 32, 48, 64):
                delta = (scale * candidate[:, label]).clamp(-0.08, 0.08)
                for cutoff in cutoff_values:
                    selected = utility[:, label] >= cutoff
                    if float(selected.float().mean()) > 0.5 or int(selected.sum()) == 0:
                        continue
                    benefit = float((((sign * delta) > 0) & selected).sum() / selected.sum())
                    if benefit < 0.65:
                        continue
                    routed = base[:, label] + selected * delta
                    gains = {
                        name: label_f1(routed[index], target[index, label], thresholds[label]) - base_group[name]
                        for name, index in offsets.items()
                    }
                    gain_tensor = torch.tensor(list(gains.values()))
                    nondegrading = int((gain_tensor >= -0.002).sum())
                    pooled_gain = label_f1(routed, target[:, label], thresholds[label]) - label_f1(base[:, label], target[:, label], thresholds[label])
                    if pooled_gain <= 0 or nondegrading < 2:
                        continue
                    score = float(gain_tensor.mean() - 0.5 * gain_tensor.std(unbiased=False))
                    if score > best["score"]:
                        best = {"score": score, "scale": scale, "cutoff": cutoff, "source": source, "gains": gains, "pooled_gain": pooled_gain}
        test_delta = (best["scale"] * test_candidate[:, label]).clamp(-0.08, 0.08)
        test_selected = test_utilities[best["source"]][:, label] >= best["cutoff"]
        test_score = label_f1(test_base[:, label] + test_selected * test_delta, test_target[:, label], thresholds[label])
        choices.append(best)
        test_f1.append(test_score)
    result = {
        "protocol": "train-only cohort robust selection; test read after lock",
        "cohorts": sizes,
        "choices": choices,
        "test_per_action_f1": test_f1,
        "test_mf1": sum(test_f1) / 4,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
