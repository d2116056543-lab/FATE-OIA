from __future__ import annotations

import argparse
from pathlib import Path
import torch

from fate_oia.engine.train_pmcal_v2_oia import load_config, build_model, make_loader, evaluate
from fate_oia.utils.pmcal_artifacts import write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_feature_cache", action="store_true")
    ap.add_argument("--require_no_token_compression", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    loader = make_loader(cfg, args.split, int(cfg.get("training", {}).get("primary_batch_size", 9)), None, False, int(cfg.get("training", {}).get("num_workers", 4)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(model, loader, device, out, int(ckpt.get("epoch", 0)))
    write_json(out / "metrics_base_fixed.json", metrics["metrics_base_fixed"])
    write_json(out / "metrics_deploy_fixed.json", metrics["metrics_deploy_fixed"])
    write_json(out / "metrics_calibrated.json", metrics["metrics_calibrated"])
    write_json(out / "metrics_global_threshold_diag.json", metrics["metrics_global_threshold_diag"])
    write_json(out / "metrics_test_oracle_per_label_diag.json", metrics["metrics_test_oracle_per_label_diag"])
    write_json(out / "threshold_search_test_oracle_DIAGNOSTIC_ONLY.json", metrics["metrics_test_oracle_per_label_diag"])


if __name__ == "__main__":
    main()
