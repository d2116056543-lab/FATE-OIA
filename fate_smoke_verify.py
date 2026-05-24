from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset

    report = {}
    for split, expected in [("train", 16082), ("val", 2270), ("test", 4572)]:
        ds = BDDOIAMultiTaskDataset("dataset/BDD-OIA", "raw_data/BDD-OIA", split=split, action_dim=4)
        report[split] = ds.audit()
        report[split]["expected"] = expected
        report[split]["count_match"] = len(ds) == expected
    out = Path("outputs/fate_oia/preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:5000])


if __name__ == "__main__":
    main()
