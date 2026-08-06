from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .train_aie_oia import (
    build_model,
    canonical_model_state_dict,
    evaluate_epoch,
    load_config,
    make_dataset,
    make_loader,
)
from fate_oia.datasets.aie_splits import stable_split_ids
from fate_oia.utils.aie_artifacts import write_json
from fate_oia.utils.aie_hashes import aie_source_tree_sha256, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); device = torch.device(args.device)
    model = build_model(cfg, device); checkpoint = torch.load(args.checkpoint, map_location=device)
    if checkpoint.get("config_hash") != file_sha256(args.config) or checkpoint.get("source_tree_hash") != aie_source_tree_sha256():
        raise RuntimeError("Evaluation checkpoint does not match current AIE config/source tree")
    # Training checkpoints canonicalize a verified DINO attention alias on
    # save/resume. Standalone evaluation must use the identical contract.
    model.load_state_dict(canonical_model_state_dict(checkpoint["model"]), strict=True)
    train = make_dataset(cfg, "train"); test = make_dataset(cfg, "test")
    train_ids = [sample.file_name for sample in train.samples]
    split = stable_split_ids(train_ids, int(cfg["data"]["split_seed"]), float(cfg["data"]["train_calib_fraction"]), int(cfg["data"]["train_audit_count"]))
    id_to_index = {sample.file_name: index for index, sample in enumerate(train.samples)}
    calib_indices = [id_to_index[file_name] for file_name in split["train_calib"]]
    test_count = min(args.max_test_samples or len(test), len(test))
    calib_loader = make_loader(torch.utils.data.Subset(train, calib_indices), int(cfg["training"]["batch_size"]), False, int(cfg["data"]["num_workers"]), cfg)
    test_loader = make_loader(torch.utils.data.Subset(test, list(range(test_count))), int(cfg["training"]["batch_size"]), False, int(cfg["data"]["num_workers"]), cfg)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_epoch(model, calib_loader, test_loader, device, output, 1.0, 1.0, cfg); write_json(output / "eval_metrics.json", metrics)


if __name__ == "__main__": main()
