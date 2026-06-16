from __future__ import annotations

"""Standalone Candidate-Probe launcher.

This module is intentionally thin: the training implementation lives in
``train_acpr_oia`` so Stage A uses the same direct-image/test-only/no-cache
protocol as the rest of ACPR. The wrapper exists to make the Candidate-Probe
entry point explicit for audit and reproducibility.
"""

import sys

from fate_oia.engine.train_acpr_oia import main as train_main


def main() -> None:
    if "--stage_mode" not in sys.argv:
        sys.argv.extend(["--stage_mode", "candidate_probe"])
    train_main()


if __name__ == "__main__":
    main()
