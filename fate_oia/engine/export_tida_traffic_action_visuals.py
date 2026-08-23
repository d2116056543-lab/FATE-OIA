from __future__ import annotations

import argparse
import json
import math
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


def _draw_arrow(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    delta: tuple[float, float],
    color: tuple[int, int, int],
    scale: float,
) -> None:
    end = (origin[0] + scale * delta[0], origin[1] + scale * delta[1])
    draw.line((*origin, *end), fill=color, width=3)
    angle = math.atan2(end[1] - origin[1], end[0] - origin[0])
    for offset in (-0.55, 0.55):
        head = (end[0] - 8 * math.cos(angle + offset), end[1] - 8 * math.sin(angle + offset))
        draw.line((*end, *head), fill=color, width=3)


def render_patch_motion_vectors(
    raw: torch.Tensor,
    common: torch.Tensor,
    exclusive: torch.Tensor,
    path: Path,
) -> None:
    if raw.ndim != 3 or raw.shape[-1] != 2 or common.shape != (raw.shape[0], 2):
        raise ValueError("raw/common motion must be [T,A,2] and [T,2]")
    if exclusive.shape != raw.shape:
        raise ValueError("exclusive motion must match raw motion")
    raw, common, exclusive = (value.detach().float().cpu() for value in (raw, common, exclusive))
    intervals, actions = raw.shape[:2]
    cell_w, cell_h, left, top = 92, 82, 90, 42
    image = Image.new("RGB", (left + intervals * cell_w + 20, top + actions * cell_h + 42), "white")
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "Traffic patch motion: raw=blue, common=gray, action-exclusive=red", fill="black")
    scale = 90.0
    for action in range(actions):
        draw.text((8, top + action * cell_h + 28), f"action {action}", fill="black")
        for interval in range(intervals):
            origin = (left + interval * cell_w + cell_w / 2, top + action * cell_h + cell_h / 2)
            draw.ellipse((origin[0] - 2, origin[1] - 2, origin[0] + 2, origin[1] + 2), fill="black")
            _draw_arrow(draw, origin, tuple(common[interval].tolist()), (125, 125, 125), scale)
            _draw_arrow(draw, origin, tuple(raw[interval, action].tolist()), (30, 95, 210), scale)
            _draw_arrow(draw, origin, tuple(exclusive[interval, action].tolist()), (215, 55, 45), scale)
    draw.text((left, top + actions * cell_h + 10), "oldest interval -> latest interval", fill="black")
    image.save(path)


def export(epoch_dir: Path, output_dir: Path, top_k: int = 12) -> None:
    attention = torch.load(epoch_dir / "traffic_action_attention_test.pt", map_location="cpu", weights_only=True)
    delta = torch.load(epoch_dir / "traffic_action_delta_test.pt", map_location="cpu", weights_only=True)
    target = torch.load(epoch_dir / "action_target_test.pt", map_location="cpu", weights_only=True)
    motion = torch.load(epoch_dir / "traffic_motion_energy_test.pt", map_location="cpu", weights_only=True)
    same_mass = torch.load(epoch_dir / "traffic_same_action_mass_test.pt", map_location="cpu", weights_only=True)
    patch_raw_path = epoch_dir / "traffic_patch_displacement_test.pt"
    patch_common_path = epoch_dir / "traffic_patch_common_displacement_test.pt"
    patch_exclusive_path = epoch_dir / "traffic_patch_exclusive_displacement_test.pt"
    patch_effective_path = epoch_dir / "traffic_patch_effective_motion_test.pt"
    patch_available = all(path.exists() for path in (patch_raw_path, patch_common_path, patch_exclusive_path))
    if patch_available:
        patch_raw = torch.load(patch_raw_path, map_location="cpu", weights_only=True)
        patch_common = torch.load(patch_common_path, map_location="cpu", weights_only=True)
        patch_exclusive = torch.load(patch_exclusive_path, map_location="cpu", weights_only=True)
        patch_effective = (
            torch.load(patch_effective_path, map_location="cpu", weights_only=True)
            if patch_effective_path.exists() else None
        )
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
        if patch_available:
            render_patch_motion_vectors(
                patch_raw[index], patch_common[index], patch_exclusive[index],
                case_dir / "traffic_patch_motion_vectors.png",
            )
        row = {
            "sample_index": index,
            "file_name": names[index],
            "action_target": target[index].tolist(),
            "traffic_action_delta": delta[index].tolist(),
            "gt_signed_transport": signed[index].tolist(),
            "motion_energy_by_interval": motion[index].tolist(),
            "same_action_attention_mass": same_mass[index].tolist(),
            "patch_motion_available": patch_available,
        }
        if patch_available:
            row.update({
                "patch_raw_displacement": patch_raw[index].tolist(),
                "patch_common_displacement": patch_common[index].tolist(),
                "patch_exclusive_displacement": patch_exclusive[index].tolist(),
                "patch_effective_motion": None if patch_effective is None else patch_effective[index].tolist(),
            })
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
