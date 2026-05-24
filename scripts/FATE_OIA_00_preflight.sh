#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import json, torch
from fate_oia.datasets.bdd_oia_multitask import BDDOIAMultiTaskDataset
report = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}
for split, expected in [("train",16082),("val",2270),("test",4572)]:
    ds = BDDOIAMultiTaskDataset("dataset/BDD-OIA", "raw_data/BDD-OIA", split=split)
    audit = ds.audit()
    audit["expected_count"] = expected
    audit["count_match"] = len(ds) == expected
    report[split] = audit
print(json.dumps(report, indent=2))
PY
