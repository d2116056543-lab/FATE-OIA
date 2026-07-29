from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
from fate_oia.datasets.meter_dataset import fixed_meter_split_indices
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex


def audit_source_distribution(
    *,
    data_root: str,
    raw_root: str,
    bdd100k_root: str,
    schema_path: str,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    dataset = BDDOIAMultiTaskDataset(
        data_root=data_root,
        raw_root=raw_root,
        split="train",
    )
    names = [sample.file_name for sample in dataset.samples]
    split = fixed_meter_split_indices(
        names,
        audit_fraction=0.08,
        calib_fraction=0.10,
        seed=seed,
    )
    indices = split["audit"][:sample_count]
    index = METERGroundingIndex(bdd100k_root, schema_path=schema_path)
    keys = (
        "factor_anchor_valid",
        "factor_state_valid",
        "factor_present_valid",
        "factor_absent_valid",
        "factor_source_complete",
    )
    counts = {key: [0] * 21 for key in keys}
    state_positive = [0] * 21
    state_negative = [0] * 21
    for sample_index in indices:
        target = index.typed_target(names[sample_index], split="train")
        if target is None:
            continue
        for key in keys:
            counts[key] = [
                left + int(right)
                for left, right in zip(counts[key], target[key].tolist())
            ]
        for factor in range(21):
            if not bool(target["factor_state_valid"][factor]):
                continue
            state = int(target["factor_state_target"][factor])
            state_positive[factor] += int(state == 0)
            state_negative[factor] += int(state == 1)
    return {
        "sample_count": len(indices),
        "counts": counts,
        "state_positive": state_positive,
        "state_negative": state_negative,
        "eligible_20_20": [
            factor
            for factor in range(21)
            if state_positive[factor] >= 20 and state_negative[factor] >= 20
        ],
        "coverage": index.coverage([names[index] for index in indices]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--raw_root", required=True)
    parser.add_argument("--bdd100k_root", required=True)
    parser.add_argument("--schema_path", required=True)
    parser.add_argument("--sample_count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = audit_source_distribution(
        data_root=args.data_root,
        raw_root=args.raw_root,
        bdd100k_root=args.bdd100k_root,
        schema_path=args.schema_path,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
