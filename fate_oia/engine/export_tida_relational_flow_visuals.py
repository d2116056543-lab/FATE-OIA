from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from fate_oia.datasets.tida_clip_manifest import load_manifest


def _load(epoch_dir: Path, name: str) -> torch.Tensor:
    return torch.load(epoch_dir / f"{name}_test.pt", map_location="cpu", weights_only=True)


def _pixel(point: torch.Tensor, width: int, height: int) -> tuple[int, int]:
    return (
        int((float(point[0]) + 1.0) * 0.5 * (width - 1)),
        int((float(point[1]) + 1.0) * 0.5 * (height - 1)),
    )


def _target_route_overlay(
    image: Image.Image,
    trajectories: torch.Tensor,
    selected_track: int,
    control_track: int,
    title: str,
    pair_attention: torch.Tensor | None = None,
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    draw.rectangle((4, 4, min(width - 4, 520), 34), fill=(0, 0, 0))
    draw.text((12, 11), title, fill=(255, 255, 255))
    for track in range(trajectories.shape[0]):
        points = [_pixel(point, width, height) for point in trajectories[track]]
        if track == selected_track:
            color, line_width = (235, 53, 45), 5
        elif track == control_track:
            color, line_width = (130, 130, 130), 5
        else:
            color, line_width = (0, 180, 210), 2
        draw.line(points, fill=color, width=line_width)
        draw.ellipse(
            (points[-1][0] - 4, points[-1][1] - 4, points[-1][0] + 4, points[-1][1] + 4),
            fill=color,
        )
    if pair_attention is not None:
        pair = pair_attention.clone()
        pair.fill_diagonal_(0)
        for flat in pair.flatten().topk(min(3, pair.numel())).indices:
            source = int(flat // pair.shape[1])
            target = int(flat % pair.shape[1])
            if float(pair[source, target]) <= 0:
                continue
            draw.line(
                (
                    _pixel(trajectories[source, -1], width, height),
                    _pixel(trajectories[target, -1], width, height),
                ),
                fill=(255, 196, 0),
                width=3,
            )
    return overlay


def export_relational_flow_visuals(
    epoch_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    max_cases: int = 32,
) -> None:
    file_names = json.loads((epoch_dir / "file_names_test.json").read_text(encoding="utf-8"))
    records = {record.file_name.lower(): record for record in load_manifest(manifest_path)}
    xy = _load(epoch_dir, "semantic_trajectory_xy")[:, 0]
    selected = _load(epoch_dir, "relational_selected_track").long()
    random = _load(epoch_dir, "relational_random_track").long()
    predicate_ids = _load(epoch_dir, "terminal_semantic_predicate_ids").long()
    action_attention = _load(epoch_dir, "relational_action_attention")
    reason_attention = _load(epoch_dir, "relational_reason_attention")
    action_pair_attention = _load(epoch_dir, "relational_action_pair_attention")
    reason_pair_attention = _load(epoch_dir, "relational_reason_pair_attention")
    action_selected = _load(epoch_dir, "relational_action_selected_track").long()
    action_control = _load(epoch_dir, "relational_action_random_track").long()
    reason_selected = _load(epoch_dir, "relational_reason_selected_track").long()
    reason_control = _load(epoch_dir, "relational_reason_random_track").long()
    risk = _load(epoch_dir, "relational_interaction_risk")
    action_delta = _load(epoch_dir, "relational_action_delta")
    reason_delta = _load(epoch_dir, "relational_reason_delta")
    action_selected_deleted = _load(epoch_dir, "relational_action_selected_deleted_delta")
    action_control_deleted = _load(epoch_dir, "relational_action_random_deleted_delta")
    reason_selected_deleted = _load(epoch_dir, "relational_reason_selected_deleted_delta")
    reason_control_deleted = _load(epoch_dir, "relational_reason_random_deleted_delta")
    action_target = _load(epoch_dir, "action_target")
    reason_target = _load(epoch_dir, "reason_target")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for index, file_name in enumerate(file_names[: int(max_cases)]):
        record = records.get(str(file_name).lower())
        if record is None:
            raise ValueError(f"manifest record missing for {file_name}")
        case_dir = output_dir / f"{index:04d}_{Path(file_name).stem}"
        case_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(record.target_image_path).convert("RGB")
        image.save(case_dir / "original.jpg", quality=95)
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        width, height = overlay.size
        for track in range(xy.shape[1]):
            points = [
                (
                    int((float(point[0]) + 1.0) * 0.5 * (width - 1)),
                    int((float(point[1]) + 1.0) * 0.5 * (height - 1)),
                )
                for point in xy[index, track]
            ]
            color = (
                (235, 53, 45) if track == int(selected[index])
                else (120, 120, 120) if track == int(random[index])
                else (0, 180, 210)
            )
            draw.line(points, fill=color, width=4)
            draw.ellipse(
                (points[-1][0] - 5, points[-1][1] - 5, points[-1][0] + 5, points[-1][1] + 5),
                fill=color,
            )
            draw.text(points[-1], f"t{track}/p{int(predicate_ids[index, track])}", fill=color)
        pair_risk = risk[index].clone()
        pair_risk.fill_diagonal_(0)
        flat_top = pair_risk.flatten().topk(min(4, pair_risk.numel())).indices
        final_points = xy[index, :, -1]
        for flat in flat_top:
            source = int(flat // pair_risk.shape[1])
            target = int(flat % pair_risk.shape[1])
            if float(pair_risk[source, target]) <= 0:
                continue
            source_xy = (
                int((float(final_points[source, 0]) + 1) * 0.5 * (width - 1)),
                int((float(final_points[source, 1]) + 1) * 0.5 * (height - 1)),
            )
            target_xy = (
                int((float(final_points[target, 0]) + 1) * 0.5 * (width - 1)),
                int((float(final_points[target, 1]) + 1) * 0.5 * (height - 1)),
            )
            draw.line((source_xy, target_xy), fill=(255, 196, 0), width=2)
        overlay.save(case_dir / "traffic_relations.png")
        action_route_files = []
        for target_id in range(action_attention.shape[1]):
            route_name = f"action_{target_id}_traffic_route.png"
            _target_route_overlay(
                image,
                xy[index],
                int(action_selected[index, target_id]),
                int(action_control[index, target_id]),
                f"action {target_id}: red selected, gray matched control",
                action_pair_attention[index, target_id],
            ).save(case_dir / route_name)
            action_route_files.append(route_name)
        top_reasons = reason_delta[index].abs().topk(min(3, reason_delta.shape[1])).indices
        reason_route_files = []
        for target_id_tensor in top_reasons:
            target_id = int(target_id_tensor)
            route_name = f"reason_{target_id}_traffic_route.png"
            _target_route_overlay(
                image,
                xy[index],
                int(reason_selected[index, target_id]),
                int(reason_control[index, target_id]),
                f"reason {target_id}: red selected, gray matched control",
                reason_pair_attention[index, target_id],
            ).save(case_dir / route_name)
            reason_route_files.append(route_name)
        action_sign = 2.0 * action_target[index] - 1.0
        reason_sign = 2.0 * reason_target[index] - 1.0
        target_effectiveness = {
            "action_signed_margin": (action_sign * action_delta[index]).tolist(),
            "action_selected_deletion_damage": (
                action_sign * (action_delta[index] - action_selected_deleted[index])
            ).tolist(),
            "action_control_deletion_damage": (
                action_sign * (action_delta[index] - action_control_deleted[index])
            ).tolist(),
            "reason_signed_margin": (reason_sign * reason_delta[index]).tolist(),
            "reason_selected_deletion_damage": (
                reason_sign * (reason_delta[index] - reason_selected_deleted[index])
            ).tolist(),
            "reason_control_deletion_damage": (
                reason_sign * (reason_delta[index] - reason_control_deleted[index])
            ).tolist(),
        }
        summary = {
            "file_name": file_name,
            "selected_track": int(selected[index]),
            "random_control_track": int(random[index]),
            "predicate_ids_by_track": predicate_ids[index].tolist(),
            "action_attention_by_target": action_attention[index].tolist(),
            "reason_attention_by_target": reason_attention[index].tolist(),
            "action_pair_attention_by_target": action_pair_attention[index].tolist(),
            "top_reason_pair_attention": {
                str(int(target_id)): reason_pair_attention[index, int(target_id)].tolist()
                for target_id in top_reasons
            },
            "action_delta": action_delta[index].tolist(),
            "reason_delta": reason_delta[index].tolist(),
            "action_selected_track_by_target": action_selected[index].tolist(),
            "action_control_track_by_target": action_control[index].tolist(),
            "reason_selected_track_by_target": reason_selected[index].tolist(),
            "reason_control_track_by_target": reason_control[index].tolist(),
            "target_effectiveness": target_effectiveness,
            "top_pair_risk": float(pair_risk.max()),
        }
        (case_dir / "transport_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (case_dir / "report.html").write_text(
            "<html><body><h1>Relational traffic evidence</h1>"
            "<p>Red: selected necessary trajectory; gray: matched random control; "
            "cyan: other semantic tracks; yellow: high-risk interactions.</p>"
            '<img src="traffic_relations.png" style="max-width:100%">'
            + "".join(
                f'<img src="{name}" style="max-width:48%;margin:4px">'
                for name in action_route_files + reason_route_files
            )
            + f"<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>"
            "</body></html>",
            encoding="utf-8",
        )
        report_rows.append(
            f'<li><a href="{case_dir.name}/report.html">{html.escape(file_name)}</a></li>'
        )
    (output_dir / "index.html").write_text(
        "<html><body><h1>TIDA relational traffic cases</h1><ul>"
        + "".join(report_rows)
        + "</ul></body></html>",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=32)
    args = parser.parse_args()
    export_relational_flow_visuals(
        Path(args.epoch_dir), Path(args.manifest), Path(args.output_dir), max_cases=args.max_cases
    )


if __name__ == "__main__":
    main()
