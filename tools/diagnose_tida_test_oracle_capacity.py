import argparse
import json
from pathlib import Path

import torch


def load(path):
    return torch.load(path, map_location="cpu").float()


def f1(logits, target, threshold):
    pred = logits.sigmoid() >= threshold
    pos = target > 0.5
    tp = (pred & pos).sum().float()
    fp = (pred & ~pos).sum().float()
    fn = (~pred & pos).sum().float()
    return float(2 * tp / (2 * tp + fp + fn).clamp_min(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.65,0.68,0.60,0.60")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.epoch_dir
    base = load(root / "pre_object_intent_action_test.pt")
    candidate = load(root / "object_intent_action_candidate_test.pt")
    target = load(root / "action_target_test.pt")
    gates = [
        load(root / "object_intent_action_directional_utility_gate_test.pt"),
        load(root / "object_intent_action_risk_utility_gate_test.pt"),
    ]
    thresholds = torch.tensor([float(v) for v in args.thresholds.split(",")])
    rows = []
    for label in range(4):
        best = {"f1": f1(base[:, label], target[:, label], thresholds[label]), "scale": 0, "cutoff": 0, "source": 0}
        for source, gate in enumerate(gates):
            cutoff_grid = torch.unique(torch.cat([torch.arange(0.0, 1.001, 0.02), torch.quantile(gate[:, label], torch.arange(0.05, 1.0, 0.05))]))
            for scale in range(2, 129, 2):
                delta = (scale * candidate[:, label]).clamp(-0.08, 0.08)
                for cutoff in cutoff_grid:
                    routed = base[:, label] + (gate[:, label] >= cutoff) * delta
                    score = f1(routed, target[:, label], thresholds[label])
                    if score > best["f1"]:
                        best = {"f1": score, "scale": scale, "cutoff": float(cutoff), "source": source}
        rows.append(best)
    result = {"diagnostic_only_test_labels_used": True, "per_action": rows, "macro_f1": sum(row["f1"] for row in rows) / 4}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
