from __future__ import annotations

import argparse
from pathlib import Path

import torch


def describe(value, prefix: str = "root") -> None:
    if torch.is_tensor(value):
        print(
            prefix, "tensor", tuple(value.shape), value.dtype,
            "min", float(value.float().min()), "max", float(value.float().max()),
        )
    elif isinstance(value, dict):
        print(prefix, "dict", len(value), list(value)[:20])
        for key, item in list(value.items())[:12]:
            describe(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        print(prefix, type(value).__name__, len(value), value[:3])
    else:
        print(prefix, type(value).__name__, repr(value)[:300])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    describe(torch.load(args.path, map_location="cpu"))


if __name__ == "__main__":
    main()
