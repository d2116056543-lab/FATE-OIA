from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Create an explainability-audit manifest for a chosen FATE-OIA score checkpoint.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_visualizations", type=int, default=200)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "max_visualizations": args.max_visualizations,
        "status": "schema_ready",
        "boundary": "This audit does not replace raw Act/Exp score metrics.",
    }
    (out / "fate_snna_metadata.jsonl").write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "pointing_game_summary.json").write_text(json.dumps({"status": "pending_checkpoint_audit"}, indent=2), encoding="utf-8")
    (out / "deletion_sufficiency_summary.json").write_text(json.dumps({"status": "pending_checkpoint_audit"}, indent=2), encoding="utf-8")
    (out / "reason_template_examples.jsonl").write_text("", encoding="utf-8")
    print(json.dumps({"event": "explainability_audit_schema_ready", "output_dir": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
