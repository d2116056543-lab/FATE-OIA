"""One-batch real-image forward smoke for MOSAIC-TRUST v4 CREDO.

This script deliberately does not construct an optimizer or enter the
training loop. It prints stage markers so remote DINO/data stalls are visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fate_oia.engine.train_acpr_mosaic_trust_icdor import (
    build_icdor_loaders,
    build_icdor_model,
    load_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    parser.add_argument("--output_dir", default=".background_runs/mosaic_v4_forward_smoke")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for the real forward smoke")
    config = load_config(Path(args.config))
    print("stage=config_loaded", flush=True)
    _, audit_loader, _, _, _ = build_icdor_loaders(
        config,
        Path(args.output_dir) / "data",
        batch_size=1,
        num_workers=0,
        max_train_samples=1,
        max_audit_samples=1,
        max_calib_samples=1,
        max_test_samples=1,
    )
    batch = next(iter(audit_loader))
    print("stage=real_batch_loaded", batch["file_name"], flush=True)
    model = build_icdor_model(config).to(device)
    model.eval()
    print("stage=model_built", flush=True)
    images = batch["image"].to(device, non_blocking=True)
    with torch.no_grad():
        output = model(
            images,
            route_mode="shadow",
            latent_enabled=True,
            reason_route_mode="full",
            return_masks=True,
            return_diagnostics=True,
        )
    required = (
        "action_visual_logits", "action_shadow_logits", "action_final_logits",
        "reason_visual_logits", "reason_final_logits", "cV", "cV_ema",
        "factor_soft_masks", "factor_coarse_masks", "sampling_coordinates",
        "sampled_features", "sample_attention",
    )
    missing = [name for name in required if not isinstance(output.get(name), torch.Tensor)]
    if missing:
        raise RuntimeError(f"missing forward outputs: {missing}")
    delta = (output["factor_soft_masks"] - output["factor_coarse_masks"]).abs().mean()
    action_equal = torch.allclose(output["action_final_logits"], output["action_visual_logits"], atol=1e-7, rtol=0.0)
    print(
        "stage=forward_complete "
        f"action_shape={tuple(output['action_final_logits'].shape)} "
        f"reason_shape={tuple(output['reason_final_logits'].shape)} "
        f"cV_mean={float(output['cV'].mean()):.6f} "
        f"fine_coarse_delta={float(delta):.6f} "
        f"shadow_final_visual_equal={action_equal}",
        flush=True,
    )


if __name__ == "__main__":
    main()
