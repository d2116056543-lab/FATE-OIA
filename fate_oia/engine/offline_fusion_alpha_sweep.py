from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import torch

from fate_oia.metrics import multilabel_metrics_from_logits


def _load_tensor(path: str | Path) -> torch.Tensor:
    return torch.load(Path(path), map_location="cpu")


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def run_alpha_sweep(
    *,
    visual_action_logits: torch.Tensor,
    reason_action_logits: torch.Tensor,
    labels_action: torch.Tensor,
    reason_logits: torch.Tensor,
    labels_reason: torch.Tensor,
    output_dir: str | Path,
    action_dim: int = 4,
    alphas: Iterable[float] | None = None,
    min_gain: float = 0.003,
) -> dict:
    if visual_action_logits.shape != reason_action_logits.shape:
        raise ValueError("visual_action_logits and reason_action_logits must have the same shape")
    if visual_action_logits.shape != labels_action.shape:
        raise ValueError("action logits and labels must have the same shape")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    alpha_values = list(alphas) if alphas is not None else [round(i / 20.0, 2) for i in range(21)]
    reason_metrics = multilabel_metrics_from_logits(reason_logits, labels_reason, 0.5, prefix="Exp_")
    exp_mf1 = float(reason_metrics.get("Exp_mF1", 0.0))
    rows: list[dict] = []
    for alpha in alpha_values:
        a = float(alpha)
        logits = a * visual_action_logits + (1.0 - a) * reason_action_logits
        metrics = multilabel_metrics_from_logits(logits, labels_action, 0.5, prefix="Act_")
        joint = 0.5 * float(metrics["Act_mF1"]) + 0.5 * exp_mf1
        rows.append(
            {
                "alpha": a,
                "Act_mF1": float(metrics["Act_mF1"]),
                "Act_oF1": float(metrics["Act_oF1"]),
                "Act_mAP": float(metrics["Act_mAP"]),
                "Exp_mF1_current": exp_mf1,
                "joint": float(joint),
            }
        )
    best = max(rows, key=lambda r: r["joint"])
    alpha0 = min(rows, key=lambda r: abs(r["alpha"] - 0.0))
    gain = float(best["joint"] - alpha0["joint"])
    fusion_fix = bool(best["alpha"] > 0.0 and gain >= float(min_gain))
    result = {
        "action_dim": int(action_dim),
        "rows": rows,
        "best_alpha": float(best["alpha"]),
        "best_joint_alpha": float(best["joint"]),
        "joint_at_alpha0": float(alpha0["joint"]),
        "fusion_alpha_gain": gain,
        "fusion_fix_recommended": fusion_fix,
    }
    (output / "alpha_sweep_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "alpha_sweep_test.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline alpha sweep between visual-action and reason-action logits.")
    ap.add_argument("--run_dir", default="", help="Run/epoch directory containing saved logits.")
    ap.add_argument("--visual_action_logits", default="")
    ap.add_argument("--reason_action_logits", default="")
    ap.add_argument("--labels_action", default="")
    ap.add_argument("--reason_logits", default="")
    ap.add_argument("--labels_reason", default="")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--action_dim", type=int, default=4)
    args = ap.parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else None
    def pick(explicit: str, name: str) -> Path:
        if explicit:
            return Path(explicit)
        if run_dir is None:
            raise ValueError(f"Missing --{name} or --run_dir")
        p = run_dir / name
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    result = run_alpha_sweep(
        visual_action_logits=_load_tensor(pick(args.visual_action_logits, "logits_action_visual_test.pt")),
        reason_action_logits=_load_tensor(pick(args.reason_action_logits, "logits_action_reason_test.pt")),
        labels_action=_load_tensor(pick(args.labels_action, "labels_action_test.pt")),
        reason_logits=_load_tensor(pick(args.reason_logits, "logits_reason_test.pt")),
        labels_reason=_load_tensor(pick(args.labels_reason, "labels_reason_test.pt")),
        output_dir=args.output_dir,
        action_dim=args.action_dim,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()

