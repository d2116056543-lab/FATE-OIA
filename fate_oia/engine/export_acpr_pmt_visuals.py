from __future__ import annotations

import argparse
from pathlib import Path
from fate_oia.utils.acpr_pmt_visualization import export_chain_case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    export_chain_case(Path(args.output_dir), {
        "action": "forward",
        "reason_id": 0,
        "reason_name": "traffic or object context",
        "predicate": "front_vehicle",
        "patch_coordinates": [[22, 40]],
    })


if __name__ == "__main__":
    main()
