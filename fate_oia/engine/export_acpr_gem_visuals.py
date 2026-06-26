from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.utils.acpr_gem_artifacts import build_evidence_chain, write_evidence_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    chain = build_evidence_chain("forward", "front_object", "front_vehicle_close", "obstacle: vehicle", (40, 22))
    write_evidence_report(out / "evidence_report.html", [chain])


if __name__ == "__main__":
    main()
