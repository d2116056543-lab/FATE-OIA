from __future__ import annotations

import argparse
from pathlib import Path
from fate_oia.engine.eagle_pu_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "eagle_pu_visual_export_schema.json", {"required": ["state_attention", "prototype_transport", "state_graph", "calibration", "evidence_audit"], "available": False})

if __name__ == "__main__":
    main()
