from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .train_aie_cert_oia import build_model, evaluate, load_config, make_dataset, make_loader
from fate_oia.utils.aie_cert_calibration import AIECertCalibrationGuard
from fate_oia.utils.aie_cert_artifacts import write_json
from fate_oia.utils.aie_cert_schedule import schedule_values


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--device", default="cuda"); parser.add_argument("--max-test-samples", type=int)
    args = parser.parse_args(); cfg = load_config(args.config); device = torch.device(args.device)
    model = build_model(cfg, device); checkpoint = torch.load(args.checkpoint, map_location=device); model.load_state_dict(checkpoint["model"])
    dataset = make_dataset(cfg, "test")
    if args.max_test_samples: dataset = torch.utils.data.Subset(dataset, range(min(args.max_test_samples, len(dataset))))
    loader = make_loader(dataset, cfg["training"]["batch_size"], False, cfg["data"]["num_workers"], cfg)
    train = make_dataset(cfg, "train"); count=max(1,int(len(train)*cfg["data"]["train_calib_fraction"]))
    calib_loader=make_loader(torch.utils.data.Subset(train,range(len(train)-count,len(train))),cfg["training"]["batch_size"],False,cfg["data"]["num_workers"],cfg)
    calibration=AIECertCalibrationGuard(25); calibration.load_state_dict(checkpoint["calibration_guard"])
    schedule = schedule_values(checkpoint["optimizer_update"], checkpoint["schedule_total_updates"], cfg)
    metrics, tensors = evaluate(model, calib_loader, loader, device, schedule, calibration, cfg); root = Path(args.output_dir); root.mkdir(parents=True, exist_ok=True)
    write_json(root / "metrics.json", metrics)
    for key, value in tensors.items(): torch.save(value, root / f"{key}.pt")


if __name__ == "__main__": main()
