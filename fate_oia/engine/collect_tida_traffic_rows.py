from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fate_oia.engine.fit_tida_traffic_boundary_cv import _collect
from fate_oia.engine.train_tida_oia import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", default="online")
    parser.add_argument("--object-track-store")
    parser.add_argument("--frame-store-root")
    parser.add_argument("--partition", choices=("train_core", "train_calib", "train_audit"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--context-chunk-size", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite traffic rows: {output}")
    runtime = build_runtime(args, evaluation_only=True)
    rows = _collect(runtime.model, runtime.loaders[args.partition], runtime.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"partition": args.partition, "rows": rows}, output)
    print({
        "output": str(output), "partition": args.partition,
        "rows": len(rows["file_names"]), "unique_names": len(set(rows["file_names"])),
    }, flush=True)


if __name__ == "__main__":
    main()
