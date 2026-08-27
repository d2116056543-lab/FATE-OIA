from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fate_oia.datasets.tida_clip_manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    groups = defaultdict(list)
    for row in rows:
        groups[(row.partition, row.source_batch)].append(row)
        groups[(row.partition, "__all__")].append(row)
    output = {}
    for (partition, source), values in sorted(groups.items()):
        action = np.asarray([row.action for row in values], dtype=np.float64)
        reason = np.asarray([row.reason for row in values], dtype=np.float64)
        output[f"{partition}/{source}"] = {
            "count": len(values),
            "action_positive_rate": action.mean(0).tolist(),
            "action_cardinality_mean": float(action.sum(1).mean()),
            "reason_cardinality_mean": float(reason.sum(1).mean()),
        }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
