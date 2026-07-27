"""CLI entrypoint for P20 RAEL runtime profiling."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable

from fate_oia.utils.rael_runtime import NoEligibleRuntimeProfile, profile_runtime


def _load_factory(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--runner-factory must use module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError("--runner-factory must resolve to a callable")
    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile real RAEL trainer optimizer updates.")
    parser.add_argument("--runner-factory", required=True, help="Real RAEL adapter factory as module:function")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        profile_runtime(runner_factory=_load_factory(args.runner_factory), output_dir=args.output_dir, device=args.device)
    except (NoEligibleRuntimeProfile, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
