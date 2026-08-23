from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def attention_to_time_action_grid(attention: torch.Tensor, num_actions: int = 4) -> torch.Tensor:
    if attention.ndim != 2 or attention.shape[-1] % int(num_actions):
        raise ValueError("attention must be [target_action,time*source_action]")
    return attention.reshape(attention.shape[0], -1, int(num_actions))


def _heat_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, float(value)))
    return (int(245 * value + 20), int(80 + 120 * (1.0 - value)), int(235 * (1.0 - value) + 20))


def _render_grid(grid: torch.Tensor, target_action: int, path: Path) -> None:
    values = grid[target_action].detach().float().cpu()
    scale = values.max().clamp_min(1e-8)
    values = values / scale
    cell, left, top = 28, 105, 38
    image = Image.new("RGB", (left + values.shape[0] * cell + 12, top + values.shape[1] * cell + 42), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), f"target action {target_action}: time ->, source action down", fill="black")
    for time_index in range(values.shape[0]):
        for source_action in range(values.shape[1]):
            x0, y0 = left + time_index * cell, top + source_action * cell
            draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=_heat_color(values[time_index, source_action]))
    for source_action in range(values.shape[1]):
        draw.text((8, top + source_action * cell + 7), f"source {source_action}", fill="black")
    draw.text((left, top + values.shape[1] * cell + 8), "oldest history", fill="black")
    draw.text((left + max(0, values.shape[0] - 4) * cell, top + values.shape[1] * cell + 8), "latest", fill="black")
    image.save(path)


def export(epoch_dir: Path, output_dir: Path, top_k: int = 12) -> None:
    attention = torch.load(epoch_dir / "traffic_action_attention_test.pt", map_location="cpu", weights_only=True)
    delta = torch.load(epoch_dir / "traffic_action_delta_test.pt", map_location="cpu", weights_only=True)
    target = torch.load(epoch_dir / "action_target_test.pt", map_location="cpu", weights_only=True)
    motion = torch.load(epoch_dir / "traffic_motion_energy_test.pt", map_location="cpu", weights_only=True)
    same_mass = torch.load(epoch_dir / "traffic_same_action_mass_test.pt", map_location="cpu", weights_only=True)
    names = json.loads((epoch_dir / "file_names_test.json").read_text(encoding="utf-8"))
    signed = (2.0 * target - 1.0) * delta
    score = signed.abs().max(-1).values
    selected = score.topk(min(int(top_k), score.numel())).indices.tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for rank, index in enumerate(selected):
        case_dir = output_dir / f"case_{rank:02d}_{index:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        grid = attention_to_time_action_grid(attention[index])
        for action in range(grid.shape[0]):
            _render_grid(grid, action, case_dir / f"action_{action}_traffic_attention.png")
        row = {
            "sample_index": index,
            "file_name": names[index],
            "action_target": target[index].tolist(),
            "traffic_action_delta": delta[index].tolist(),
            "gt_signed_transport": signed[index].tolist(),
            "motion_energy_by_interval": motion[index].tolist(),
            "same_action_attention_mass": same_mass[index].tolist(),
        }
        (case_dir / "traffic_transport.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        summary.append(row)
    (output_dir / "traffic_transport_top_cases.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()
    export(args.epoch_dir, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
