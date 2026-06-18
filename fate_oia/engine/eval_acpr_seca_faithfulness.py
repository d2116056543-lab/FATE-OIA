from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


EVAL_ONLY = True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_dir", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    epoch_dir = Path(args.epoch_dir)
    output = Path(args.output) if args.output else epoch_dir / "seca_faithfulness_eval.json"
    logits_action = _load(epoch_dir / "logits_action_deploy_test.pt")
    logits_reason = _load(epoch_dir / "logits_reason_deploy_test.pt")
    labels_action = _load(epoch_dir / "labels_action_test.pt")
    labels_reason = _load(epoch_dir / "labels_reason_test.pt")
    attn = _load(epoch_dir / "seca_action_reason_attention_test.pt").float()
    if attn.numel() == 0:
        result = {"available": False, "reason": "missing SECA attention"}
    else:
        action_prob = torch.sigmoid(logits_action)
        reason_prob = torch.sigmoid(logits_reason)
        no_null = attn[..., :21]
        selected = no_null.topk(k=min(args.topk, no_null.shape[-1]), dim=-1).indices
        selected_scores = torch.gather(reason_prob.unsqueeze(1).expand(-1, 4, -1), 2, selected).mean(dim=-1)
        rolled = torch.roll(selected, shifts=1, dims=-1)
        random_scores = torch.gather(reason_prob.unsqueeze(1).expand(-1, 4, -1), 2, rolled).mean(dim=-1)
        pos_action_mask = labels_action > 0.5
        selected_pos = selected_scores[pos_action_mask].mean() if pos_action_mask.any() else torch.tensor(0.0)
        random_pos = random_scores[pos_action_mask].mean() if pos_action_mask.any() else torch.tensor(0.0)
        result = {
            "available": True,
            "method": "eval_only_artifact_selected_vs_deterministic_random_reason_support",
            "eval_only": EVAL_ONLY,
            "sample_count": int(logits_action.shape[0]),
            "topk": int(args.topk),
            "selected_reason_score_positive_action_mean": float(selected_pos),
            "random_reason_score_positive_action_mean": float(random_pos),
            "selected_minus_random": float(selected_pos - random_pos),
            "action_positive_rate": float(labels_action.mean()),
            "reason_positive_rate": float(labels_reason.mean()),
            "note": "Eval-only diagnostic from saved logits/attention; no gradients or parameter updates.",
        }
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
