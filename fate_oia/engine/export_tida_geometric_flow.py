from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fate_oia.engine.train_tida_oia import build_runtime


def _device_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return ((tensor.float() * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def export(args) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    runtime = build_runtime(args, evaluation_only=True)
    runtime.model.eval()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    exported = 0
    for batch in runtime.loaders["test"]:
        batch = _device_batch(batch, runtime.device)
        output = runtime.model(
            batch["target_image"], batch["context_images"], batch["timestamps"], batch["frame_valid_mask"],
            temporal_action_scale=1.0, temporal_reason_scale=1.0,
        )
        for index, file_name in enumerate(batch["file_name"]):
            case_dir = output_root / f"{exported:03d}_{Path(file_name).stem}"
            case_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_rgb(batch["context_images"][index, -1])).save(case_dir / "last_history_frame.jpg")
            flow = output["geometric_flow_field"][index].float().cpu()
            mean_flow = flow.mean(0)
            energy = flow.square().sum(1).sqrt().mean(0).numpy()
            plt.imsave(case_dir / "traffic_flow_energy.png", energy, cmap="inferno")
            stride = 4
            y, x = np.mgrid[0 : mean_flow.shape[-2] : stride, 0 : mean_flow.shape[-1] : stride]
            fig, axis = plt.subplots(figsize=(10, 5.6), dpi=120)
            axis.imshow(energy, cmap="gray")
            axis.quiver(
                x, y, mean_flow[0, ::stride, ::stride].numpy(), mean_flow[1, ::stride, ::stride].numpy(),
                color="#00d8ff", angles="xy", scale_units="xy", scale=0.35,
            )
            axis.set_axis_off(); fig.tight_layout(pad=0)
            fig.savefig(case_dir / "traffic_flow_quiver.png", bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            trace = {
                "file_name": file_name,
                "history_fractions": [0.25, 0.50, 0.75, 1.0],
                "action_prefix_logits": output["prefix_video_action_logits"][index].float().cpu().tolist(),
                "reason_prefix_logits": output["prefix_video_reason_logits"][index].float().cpu().tolist(),
                "geometric_action_delta": output["geometric_action_delta"][index].float().cpu().tolist(),
                "geometric_reason_delta": output["geometric_reason_delta_effective"][index].float().cpu().tolist(),
                "global_horizontal_by_step": output["geometric_global_horizontal"][index].float().cpu().tolist(),
                "global_expansion_by_step": output["geometric_global_expansion"][index].float().cpu().tolist(),
                "region_motion_by_step": output["geometric_region_motion"][index].float().cpu().tolist(),
            }
            (case_dir / "traffic_flow_target_transport.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (case_dir / "report.html").write_text(
                "<html><body><h1>TIDA Geometric Traffic Flow</h1>"
                "<img src='last_history_frame.jpg' width='48%'><img src='traffic_flow_quiver.png' width='48%'>"
                "<p>Exact per-target temporal contributions are stored in traffic_flow_target_transport.json.</p>"
                "</body></html>", encoding="utf-8"
            )
            exported += 1
            if exported >= args.max_cases:
                return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-view", choices=("online", "ema"), default="ema")
    parser.add_argument("--clip-manifest", required=True)
    parser.add_argument("--image-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=16)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
