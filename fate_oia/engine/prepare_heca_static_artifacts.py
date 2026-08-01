from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
import yaml

from fate_oia.datasets.meter_dataset import METERDataset, fixed_meter_split_indices
from fate_oia.datasets.meter_grounding_index import METERGroundingIndex
from fate_oia.datasets.meter_typed_targets import compute_factor_observability_tau
from fate_oia.models.meter_schema import METERFactorSchema
from fate_oia.transforms_meter import meter_image_transform


def build_tau_from_train_main(
    file_names: list[str],
    main_indices: list[int],
    target_provider: Callable[[str], dict[str, torch.Tensor]],
    factor_groups: list[str],
) -> tuple[torch.Tensor, dict[str, object]]:
    observed = torch.zeros(len(factor_groups))
    valid = torch.zeros(len(factor_groups))
    for index in main_indices:
        target = target_provider(file_names[index])
        target_valid = target["factor_observability_valid"].float()
        observed += target["factor_observability"].float() * target_valid
        valid += target_valid
    tau = compute_factor_observability_tau(observed, valid, factor_groups, alpha=20.0)
    names = [file_names[index] for index in main_indices]
    metadata = {
        "fit_split": "train_main",
        "source_split": "train_main",
        "sample_count": len(main_indices),
        "sample_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        "alpha": 20.0,
        "observed_count": observed.tolist(),
        "valid_count": valid.tolist(),
        # This is source eligibility for train-time weak supervision, not a
        # target that the image model is allowed to predict at test time.
        "provenance_valid_count": valid.tolist(),
        "provenance_semantics": "train_only_weak_source_eligibility",
        # Kept solely for legacy artifact readers; this statistic is never
        # loaded by METEROIAModel or used as a learning/calibration threshold.
        "legacy_tau_not_used_for_model": True,
        "tau": tau.tolist(),
    }
    return tau, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    schema_path = Path("configs/meter_factor_schema.yaml")
    schema = METERFactorSchema(schema_path)
    dataset = METERDataset(
        data_root=config["data"]["data_root"],
        raw_root=config["data"]["raw_root"],
        split="train",
        transform=meter_image_transform(),
    )
    names = [sample.file_name for sample in dataset.base.samples]
    split = fixed_meter_split_indices(
        names,
        audit_fraction=float(config["splits"]["audit_fraction"]),
        calib_fraction=float(config["splits"]["calib_fraction"]),
        seed=int(config["splits"]["seed"]),
    )
    grounding = METERGroundingIndex(
        config["data"]["bdd100k_root"], schema_path=schema_path
    )
    tau, metadata = build_tau_from_train_main(
        names,
        split["main"],
        lambda name: grounding.typed_target(name, split="train"),
        [str(row["factor_group"]) for row in schema.rows],
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(tau, output / "factor_observability_tau.pt")
    (output / "factor_source_statistics.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "factor_provenance_stats.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "factor_observability_tau_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "heca_tau_stats.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
