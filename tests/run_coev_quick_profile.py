from __future__ import annotations

import argparse
from pathlib import Path

from fate_oia.engine.train_coev_oia import load_config
from fate_oia.utils.coev_preflight import memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--accum", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["num_workers"] = args.workers
    cfg["model"]["history_chunk_size"] = args.chunk
    cfg["backbone"]["activation_checkpointing"] = args.checkpointing
    cfg["memory_probe"]["candidates"] = [[args.batch, args.accum]]
    cfg["memory_probe"]["warmup_updates"] = 0
    cfg["memory_probe"]["measured_updates"] = args.updates
    cfg["memory_probe"]["stress_updates"] = args.updates
    print(memory(cfg, Path(args.output)))


if __name__ == "__main__":
    main()
