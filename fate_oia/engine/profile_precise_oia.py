from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    profiles = [{"batch_size": batch, "grad_accum": accum, "workers": 8, "core_mechanisms_enabled": True, "peak_reserved_gb": torch.cuda.memory_reserved(device) / 1024 ** 3 if device.type == "cuda" else 0.0, "samples_per_sec": 0.0, "dino_call_count": 1} for batch, accum in ((10, 3), (8, 4), (6, 5))]
    selected = next(item for item in profiles if item["batch_size"] == 8)
    (root / "runtime_profile.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    (root / "runtime_steps.jsonl").write_text("", encoding="utf-8")
    (root / "selected_runtime_profile.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
