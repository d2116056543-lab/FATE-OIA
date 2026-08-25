from pathlib import Path
import json
import torch


ROOT = Path(r"F:\FATE_Drive_runs\tida_object_role_grounded_v8_8_pilot1000\epoch_000")
THRESHOLDS = torch.tensor([0.59000003, 0.65999997, 0.60000002, 0.61000001])
SCALES = torch.tensor([-4.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0])


def f1(logits, target, thresholds):
    pred = logits.sigmoid() >= thresholds
    truth = target > 0.5
    tp = (pred & truth).sum(0).float()
    fp = (pred & ~truth).sum(0).float()
    fn = (~pred & truth).sum(0).float()
    return 2 * tp / (2 * tp + fp + fn).clamp_min(1)


base = torch.load(ROOT / "pre_object_intent_action_test.pt", weights_only=True)
delta = torch.load(ROOT / "object_intent_action_candidate_test.pt", weights_only=True)
target = torch.load(ROOT / "action_target_test.pt", weights_only=True)
best_scale, best_f1 = [], []
for label in range(4):
    values = []
    for scale in SCALES:
        values.append(float(f1(base[:, label:label+1] + scale * delta[:, label:label+1], target[:, label:label+1], THRESHOLDS[label:label+1])[0]))
    index = int(torch.tensor(values).argmax())
    best_scale.append(float(SCALES[index]))
    best_f1.append(values[index])
gain = torch.tensor(best_scale)
final = base + delta * gain
result = {
    "baseline_f1": f1(base, target, THRESHOLDS).tolist(),
    "scale_grid": SCALES.tolist(),
    "test_oracle_best_scale_by_label": best_scale,
    "test_oracle_best_f1_by_label": best_f1,
    "test_oracle_mf1": float(f1(final, target, THRESHOLDS).mean()),
    "warning": "diagnostic ceiling only; test labels must never fit deployment gains",
}
print(json.dumps(result, indent=2))
