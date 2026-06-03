from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.psr_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect standardized PSR logits by direct-image test-only eval when a checkpoint lacks logits.")
    ap.add_argument("--candidate_name", required=True)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--config_or_args", default="")
    ap.add_argument("--bdd_oia_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--test_only", action="store_true")
    ap.add_argument("--no_feature_cache", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "collect_manifest.json", vars(args) | {"status": "not_collected", "reason": "No checkpoint wrapper was required for current discovered candidates."})
    raise NotImplementedError("Direct-image PSR logit collection is only used if required logits are missing; current implementation records a clear missing-candidate state instead of fabricating logits.")


if __name__ == "__main__":
    main()
