from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

ACTION_NAMES_4 = ["forward", "stop", "left", "right"]
DEFAULT_REASON_NAMES = [
    "traffic light is red", "traffic light is green", "traffic sign", "front vehicle", "pedestrian",
    "rider or bike", "parked vehicle", "obstacle", "lane change or turn left", "lane change or turn right",
    "clear road", "crosswalk", "solid lane line", "road curb", "drivable area", "vehicle ahead slows",
    "oncoming traffic", "intersection", "weather or visibility", "uncertain scene", "other driving reason",
]


def _load_tensor(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")


def _load_names(path: str | Path | None, count: int) -> list[str]:
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        values = data.get("reason_names") or data.get("names") or [] if isinstance(data, dict) else data
        names = [str(x) for x in values]
    else:
        names = DEFAULT_REASON_NAMES[:]
    while len(names) < count:
        names.append(f"reason_{len(names)}")
    return names[:count]


def render_from_tensors(
    action_logits: torch.Tensor,
    reason_logits: torch.Tensor,
    labels: torch.Tensor,
    file_names: Sequence[str],
    action_dim: int,
    threshold: float,
    output_path: str | Path,
    reason_names: Sequence[str] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    action_names = ACTION_NAMES_4[:action_dim]
    reason_count = int(reason_logits.shape[1]) if reason_logits.ndim == 2 else max(int(labels.shape[1]) - action_dim, 0)
    names = list(reason_names) if reason_names is not None else DEFAULT_REASON_NAMES[:]
    while len(names) < reason_count:
        names.append(f"reason_{len(names)}")
    action_probs = torch.sigmoid(action_logits.float()) if action_logits.numel() else torch.empty(0, action_dim)
    reason_probs = torch.sigmoid(reason_logits.float()) if reason_logits.numel() else torch.empty(0, reason_count)
    labels = labels.float()
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(int(action_probs.shape[0])):
            pred_action_idx = torch.where(action_probs[i] >= threshold)[0].tolist()
            pred_reason_idx = torch.where(reason_probs[i] >= threshold)[0].tolist() if reason_probs.numel() else []
            gt_action_idx = torch.where(labels[i, :action_dim] > 0)[0].tolist() if labels.numel() else []
            gt_reason_idx = torch.where(labels[i, action_dim:] > 0)[0].tolist() if labels.numel() else []
            pred_actions = [action_names[j] for j in pred_action_idx]
            pred_reasons = [names[j] for j in pred_reason_idx]
            action_text = ", ".join(pred_actions) if pred_actions else "keep uncertain action"
            reason_text = ", ".join(pred_reasons) if pred_reasons else "no high-confidence structured reason"
            row = {
                "file_name": str(file_names[i]) if i < len(file_names) else str(i),
                "pred_actions": pred_actions,
                "pred_reason_indices": pred_reason_idx,
                "pred_reason_names": pred_reasons,
                "template_explanation_en": f"The vehicle should {action_text} because {reason_text}.",
                "template_explanation_zh": f"车辆应执行 {action_text}，因为 {reason_text}。",
                "gt_actions": [action_names[j] for j in gt_action_idx],
                "gt_reason_indices": gt_reason_idx,
                "scores": {
                    "action": {action_names[j]: float(action_probs[i, j]) for j in range(action_dim)},
                    "reason": {names[j]: float(reason_probs[i, j]) for j in range(reason_count)},
                },
                "note": "BDD-OIA has structured 21-way reason labels; this is template verbalization, not free-form generation.",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render BDD-OIA action/reason logits as structured template explanations.")
    ap.add_argument("--action_logits", required=True)
    ap.add_argument("--reason_logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--file_names", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--reason_names", default="")
    args = ap.parse_args()
    file_names = json.loads(Path(args.file_names).read_text(encoding="utf-8"))
    action_logits = _load_tensor(args.action_logits)
    reason_logits = _load_tensor(args.reason_logits)
    labels = _load_tensor(args.labels)
    names = _load_names(args.reason_names, reason_logits.shape[1])
    render_from_tensors(action_logits, reason_logits, labels, file_names, args.action_dim, args.threshold, args.output, names)
    print(json.dumps({"output": args.output, "rows": int(action_logits.shape[0])}))


if __name__ == "__main__":
    main()
