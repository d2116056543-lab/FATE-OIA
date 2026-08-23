from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fate_oia.engine.train_tida_oia import build_runtime


ACTION_NAMES = ("forward", "stop", "left", "right")
ACTION_COLORS = ("#00c2ff", "#ff4d3d", "#ffd21f", "#22d878")


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _rgb(tensor: torch.Tensor) -> np.ndarray:
    mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    return ((tensor.float() * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def trajectory_case_trace(output: dict[str, torch.Tensor], index: int, file_name: str) -> dict[str, Any]:
    image_logits = output["image_action_logits"][index].float().detach().cpu()
    delta = output["traffic_trajectory_delta"][index].float().detach().cpu()
    actions = []
    for action, name in enumerate(ACTION_NAMES):
        tracks = []
        for track in range(output["trajectory_xy"].shape[2]):
            tracks.append({
                "track_id": track,
                "attention": float(output["trajectory_attention"][index, action, track]),
                "cycle_confidence_mean": float(
                    output["trajectory_cycle_confidence"][index, action, track].float().mean()
                ),
                "xy": output["trajectory_xy"][index, action, track].float().detach().cpu().tolist(),
                "speed": output["trajectory_speed"][index, action, track].float().detach().cpu().tolist(),
                "acceleration": output["trajectory_acceleration"][index, action, track].float().detach().cpu().tolist(),
                "radial_motion": output["trajectory_radial_motion"][index, action, track].float().detach().cpu().tolist(),
            })
        actions.append({
            "action_id": action, "action_name": name,
            "trust": float(output["traffic_trajectory_trust"][index, action]),
            "support": float(output["traffic_trajectory_support"][index, action]),
            "logit_delta": float(delta[action]), "tracks": tracks,
        })
    return {
        "file_name": file_name,
        "image_action_logits": image_logits.tolist(),
        "trajectory_delta": delta.tolist(),
        "trajectory_action_logits": (image_logits + delta).tolist(),
        "full_video_action_logits": output["video_action_logits"][index].float().detach().cpu().tolist(),
        "actions": actions,
    }


def _draw_action_trajectories(image: np.ndarray, xy: torch.Tensor, attention: torch.Tensor, action: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    height, width = image.shape[:2]
    fig, axis = plt.subplots(figsize=(10, 5.625), dpi=120)
    axis.imshow(image)
    for track in range(xy.shape[1]):
        points = xy[action, track].float().cpu().numpy()
        px = (points[:, 0] + 1.0) * 0.5 * (width - 1)
        py = (points[:, 1] + 1.0) * 0.5 * (height - 1)
        alpha = float(np.clip(attention[action, track].item(), 0.15, 1.0))
        axis.plot(px, py, color=ACTION_COLORS[action], linewidth=1.0 + 4.0 * alpha, alpha=alpha)
        axis.scatter(px[-1], py[-1], color=ACTION_COLORS[action], s=25 + 70 * alpha, edgecolors="black")
    axis.set_title(f"{ACTION_NAMES[action]}: action-attentive cycle-consistent traffic trajectories")
    axis.set_axis_off()
    fig.tight_layout(pad=0.25)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def export(args: argparse.Namespace) -> None:
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
        controls = {
            name: runtime.model.rerun_temporal_from_output(
                output, name, temporal_action_scale=1.0, temporal_reason_scale=1.0
            )
            for name in ("time_shuffle", "time_reverse", "repeated_last", "history_off")
        }
        for index, file_name in enumerate(batch["file_name"]):
            case_dir = output_root / f"{exported:03d}_{Path(file_name).stem}"
            case_dir.mkdir(parents=True, exist_ok=True)
            target = _rgb(batch["target_image"][index])
            Image.fromarray(target).save(case_dir / "target_frame.jpg")
            for action, name in enumerate(ACTION_NAMES):
                _draw_action_trajectories(
                    target, output["trajectory_xy"][index], output["trajectory_attention"][index],
                    action, case_dir / f"trajectory_{name}.png",
                )
            trace = trajectory_case_trace(output, index, file_name)
            sign = 2.0 * batch["action"][index].float() - 1.0
            trace["ordered_vs_control_gt_margin"] = {
                name: (sign * (
                    output["video_action_logits"][index] - changed["video_action_logits"][index]
                )).float().cpu().tolist()
                for name, changed in controls.items()
            }
            (case_dir / "trajectory_target_transport.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            images = "".join(
                f"<h2>{name}</h2><img src='trajectory_{name}.png' width='96%'>" for name in ACTION_NAMES
            )
            (case_dir / "report.html").write_text(
                "<html><body><h1>TIDA Trajectory-Relational Traffic Credit</h1>"
                "<p>Line width is target-conditioned trajectory attention; the endpoint is the target-frame anchor.</p>"
                + images
                + "<p>Exact transport and ordered-vs-control margins are in trajectory_target_transport.json.</p>"
                "</body></html>", encoding="utf-8",
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
