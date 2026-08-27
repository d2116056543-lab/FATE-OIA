from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fate_oia.datasets.tida_clip_manifest import load_manifest


def repair_names(manifest_path: Path, source: Path, output: Path) -> dict[str, object]:
    payload = torch.load(source, map_location="cpu")
    rows = payload["rows"]
    records = load_manifest(manifest_path)
    core = [record for record in records if record.partition == "train_core"]
    calib = [record for record in records if record.partition == "train_calib"]
    expected = core + calib
    if len(expected) != len(rows["action_target"]):
        raise ValueError("manifest and traffic artifact row counts differ")
    action = torch.tensor([record.action for record in expected], dtype=torch.float32)
    reason = torch.tensor([record.reason for record in expected], dtype=torch.float32)
    if not torch.equal(action, rows["action_target"]):
        raise ValueError("action labels do not match deterministic dataset order")
    if not torch.equal(reason, rows["reason_target"]):
        raise ValueError("reason labels do not match deterministic dataset order")
    sources = [record.source_batch for record in expected]
    if sources != list(rows["source_batches"]):
        raise ValueError("source batches do not match deterministic dataset order")
    names = [record.file_name for record in expected]
    if len(set(names)) != len(names):
        raise ValueError("manifest file_name values are not unique")
    rows["file_names"] = names
    payload["name_repair"] = {
        "source": str(manifest_path),
        "method": "deterministic_dataset_partition_order",
        "action_labels_exact": True,
        "reason_labels_exact": True,
        "source_batches_exact": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    return {
        "output": str(output), "rows": len(names), "core": len(core),
        "calib": len(calib), "first": names[0], "last": names[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(repair_names(args.manifest, args.source, args.output))


if __name__ == "__main__":
    main()
