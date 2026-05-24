from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def _load(path: str) -> torch.Tensor:
    p = Path(path)
    if p.suffix in {".pt", ".pth"}:
        return torch.load(p, map_location="cpu")
    return torch.tensor(json.loads(p.read_text(encoding="utf-8")))


def counterfactual_delta_summary(original_logits: torch.Tensor, masked_logits: torch.Tensor, labels: torch.Tensor) -> dict:
    labels = labels.float()
    orig = F.binary_cross_entropy_with_logits(original_logits.float(), labels, reduction="none").mean(1)
    masked = F.binary_cross_entropy_with_logits(masked_logits.float(), labels, reduction="none").mean(1)
    delta = masked - orig
    return {
        "count": int(delta.numel()),
        "mean_delta_bce": float(delta.mean().item()),
        "positive_delta_rate": float((delta > 0).float().mean().item()),
        "median_delta_bce": float(delta.median().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate counterfactual deletion impact from original/masked logits.")
    ap.add_argument("--original_logits", required=True)
    ap.add_argument("--masked_logits", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    result = counterfactual_delta_summary(_load(args.original_logits), _load(args.masked_logits), _load(args.labels))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()