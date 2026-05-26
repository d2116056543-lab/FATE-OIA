from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.models.label_cooccurrence import build_label_statistics, conditional_bias_matrix, pmi_bias_matrix


def main() -> None:
    ap = argparse.ArgumentParser(description="Build BDD-OIA action+reason co-occurrence/PMI label bias matrices.")
    ap.add_argument("--data_root", default="dataset/BDD-OIA")
    ap.add_argument("--raw_root", default="raw_data/BDD-OIA")
    ap.add_argument("--split", default="train")
    ap.add_argument("--action_dim", type=int, default=4)
    ap.add_argument("--reason_dim", type=int, default=21)
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoothing", type=float, default=1.0)
    args = ap.parse_args()
    dataset = BDDOIAMultiTaskDataset(
        data_root=args.data_root,
        raw_root=args.raw_root,
        split=args.split,
        action_dim=args.action_dim,
        reason_dim=args.reason_dim,
        load_image=False,
    )
    labels = torch.stack([torch.cat([sample["action"], sample["reason"]]) for sample in dataset])
    stats = build_label_statistics(labels, smoothing=args.smoothing)
    payload = {
        "num_samples": stats.num_samples,
        "positive_counts": stats.positive_counts.tolist(),
        "conditional_log_bias": conditional_bias_matrix(stats).tolist(),
        "pmi_bias": pmi_bias_matrix(stats).tolist(),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "num_samples": stats.num_samples}, ensure_ascii=False))


if __name__ == "__main__":
    main()

