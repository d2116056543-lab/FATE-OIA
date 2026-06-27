from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(f"<html><body>ACPR-InteractFlow atlas: {args.input_dir}</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()

