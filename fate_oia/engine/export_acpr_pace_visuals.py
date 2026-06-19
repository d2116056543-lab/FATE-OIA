from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    epoch_dir = Path(args.epoch_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    contrib = epoch_dir / "pace_action_reason_predicate_contrib_test.pt"
    available = contrib.exists()
    chains = out / "pace_evidence_chains.jsonl"
    if available:
        chains.write_text(json.dumps({"available": True, "source": str(contrib), "chain": "action -> reason -> predicate -> patch"}) + "\n", encoding="utf-8")
    else:
        chains.write_text(json.dumps({"available": False, "reason": f"missing contribution artifact: {contrib}"}) + "\n", encoding="utf-8")
    report = out / "pace_evidence_report.html"
    report.write_text(f"<html><body><h1>ACPR-PACE visual export</h1><p>available={available}</p><p>source={contrib}</p></body></html>", encoding="utf-8")
    manifest = {
        "available": available,
        "source_epoch_dir": str(epoch_dir),
        "contribution_artifact": str(contrib),
        "exports": ["pace_evidence_chains.jsonl", "pace_evidence_report.html"],
    }
    if not available:
        manifest["reason"] = f"missing contribution artifact: {contrib}"
    (out / "pace_visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
