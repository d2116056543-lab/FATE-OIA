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


def _route_image(
    image: Image.Image,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    future: torch.Tensor,
    selected: int,
    control: int,
    title: str,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    draw.rectangle((4, 4, min(width - 4, 620), 35), fill=(0, 0, 0))
    draw.text((12, 11), title, fill=(255, 255, 255))
    ego = _pixel(torch.tensor((0.0, 1.0)), width, height)
    draw.ellipse((ego[0] - 8, ego[1] - 8, ego[0] + 8, ego[1] + 8),
                 outline=(255, 255, 255), width=3)
    for track_id in range(tracks.shape[1]):
        valid = visibility[:, track_id].bool()
        points = [_pixel(point, width, height) for point in tracks[valid, track_id]]
        if len(points) < 2:
            continue
        color = (235, 53, 45) if track_id == selected else (
            (130, 130, 130) if track_id == control else (0, 180, 210)
        )
        draw.line(points, fill=color, width=5 if track_id in (selected, control) else 2)
        draw.ellipse((points[-1][0] - 4, points[-1][1] - 4,
                      points[-1][0] + 4, points[-1][1] + 4), fill=color)
        if track_id == selected:
            future_points = [points[-1]] + [
                _pixel(point, width, height) for point in future[track_id]
            ]
            draw.line(future_points, fill=(255, 196, 0), width=3)
            for point in future_points[1:]:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3),
                             fill=(255, 196, 0))
    return result


def _pair_route_image(
    image: Image.Image,
    tracks: torch.Tensor,
    visibility: torch.Tensor,
    future: torch.Tensor,
    pair_index: int,
    title: str,
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    track_count = tracks.shape[1]
    source, target = divmod(pair_index, track_count)
    draw.rectangle((4, 4, min(width - 4, 760), 35), fill=(0, 0, 0))
    draw.text((12, 11), title, fill=(255, 255, 255))
    future_pixels: dict[int, list[tuple[int, int]]] = {}
    for track_id, color in ((source, (235, 53, 45)), (target, (0, 180, 210))):
        valid = visibility[:, track_id].bool()
        history = [_pixel(point, width, height) for point in tracks[valid, track_id]]
        if not history:
            continue
        draw.line(history, fill=color, width=5)
        projected = [history[-1]] + [_pixel(point, width, height) for point in future[track_id]]
        future_pixels[track_id] = projected
        draw.line(projected, fill=color, width=3)
    if source in future_pixels and target in future_pixels:
        paired = list(zip(future_pixels[source][1:], future_pixels[target][1:]))
        if paired:
            closest = min(
                paired,
                key=lambda points: (points[0][0] - points[1][0]) ** 2
                + (points[0][1] - points[1][1]) ** 2,
            )
            draw.line(closest, fill=(255, 196, 0), width=4)
    return result


def export_object_intent_visuals(
    epoch_dir: Path, manifest_path: Path, output_dir: Path, *, max_cases: int = 32,
) -> None:
    names = json.loads((epoch_dir / "file_names_test.json").read_text(encoding="utf-8"))
    records = {record.file_name.lower(): record for record in load_manifest(manifest_path)}
    tracks = _load(epoch_dir, "object_tracks_xy")
    visibility = _load(epoch_dir, "object_tracks_visibility").bool()
    future = _load(epoch_dir, "object_intent_future_xy")
    action_attention = _load(epoch_dir, "object_intent_action_attention")
    reason_attention = _load(epoch_dir, "object_intent_reason_attention")
    action_semantic_attention = _load(
        epoch_dir, "object_intent_action_semantic_attention"
    )
    action_motion_attention = _load(epoch_dir, "object_intent_action_motion_attention")
    reason_semantic_attention = _load(
        epoch_dir, "object_intent_reason_semantic_attention"
    )
    reason_motion_attention = _load(epoch_dir, "object_intent_reason_motion_attention")
    action_motion_mix = _load(epoch_dir, "object_intent_action_motion_mix")
    reason_motion_mix = _load(epoch_dir, "object_intent_reason_motion_mix")
    action_selected = _load(epoch_dir, "object_intent_action_selected_track").long()
    action_control = _load(epoch_dir, "object_intent_action_control_track").long()
    reason_selected = _load(epoch_dir, "object_intent_reason_selected_track").long()
    reason_control = _load(epoch_dir, "object_intent_reason_control_track").long()
    action_delta = _load(epoch_dir, "object_intent_action_delta")
    reason_delta = _load(epoch_dir, "object_intent_reason_delta")
    action_selected_deleted = _load(epoch_dir, "object_intent_action_selected_deleted_delta")
    action_control_deleted = _load(epoch_dir, "object_intent_action_control_deleted_delta")
    reason_selected_deleted = _load(epoch_dir, "object_intent_reason_selected_deleted_delta")
    reason_control_deleted = _load(epoch_dir, "object_intent_reason_control_deleted_delta")
    action_target = _load(epoch_dir, "action_target")
    reason_target = _load(epoch_dir, "reason_target")
    risk = _load(epoch_dir, "object_intent_interaction_risk")
    future_ego_distance = _load(epoch_dir, "object_intent_future_ego_distance")
    future_approach_risk = _load(epoch_dir, "object_intent_future_approach_risk")
    action_pair_attention = _load(epoch_dir, "object_intent_action_pair_attention")
    reason_pair_attention = _load(epoch_dir, "object_intent_reason_pair_attention")
    action_selected_pair = _load(epoch_dir, "object_intent_action_selected_pair").long()
    action_control_pair = _load(epoch_dir, "object_intent_action_control_pair").long()
    reason_selected_pair = _load(epoch_dir, "object_intent_reason_selected_pair").long()
    reason_control_pair = _load(epoch_dir, "object_intent_reason_control_pair").long()
    action_pair_candidate = _load(epoch_dir, "object_intent_action_pair_candidate")
    reason_pair_candidate = _load(epoch_dir, "object_intent_reason_pair_candidate")
    pair_min_future_distance = _load(epoch_dir, "object_intent_pair_min_future_distance")
    pair_distance_reduction = _load(epoch_dir, "object_intent_pair_distance_reduction")
    action_utility = _load(epoch_dir, "object_intent_action_utility_gate")
    reason_utility = _load(epoch_dir, "object_intent_reason_utility_gate")
    action_utility_selected = _load(
        epoch_dir, "object_intent_action_utility_selected"
    ).bool()
    reason_utility_selected = _load(
        epoch_dir, "object_intent_reason_utility_selected"
    ).bool()
    action_deploy_scale = _load(epoch_dir, "object_intent_action_deploy_scale")
    reason_deploy_scale = _load(epoch_dir, "object_intent_reason_deploy_scale")
    action_utility_cutoff = _load(epoch_dir, "object_intent_action_utility_cutoff")
    reason_utility_cutoff = _load(epoch_dir, "object_intent_reason_utility_cutoff")
    output_dir.mkdir(parents=True, exist_ok=True)
    links = []
    for index, file_name in enumerate(names[:max_cases]):
        record = records.get(str(file_name).lower())
        if record is None:
            continue
        case_dir = output_dir / f"{index:04d}_{Path(file_name).stem}"
        case_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(record.target_image_path).convert("RGB")
        image.save(case_dir / "original.jpg", quality=95)
        route_files = []
        for target_id in range(action_delta.shape[1]):
            name = f"action_{target_id}_object_intent.png"
            _route_image(
                image, tracks[index], visibility[index], future[index],
                int(action_selected[index, target_id]), int(action_control[index, target_id]),
                f"action {target_id}: utility={float(action_utility[index, target_id]):.3f} "
                f"selected={bool(action_utility_selected[index, target_id])} "
                f"scale={float(action_deploy_scale[index, target_id]):.1f}",
            ).save(case_dir / name)
            route_files.append(name)
            pair_index = int(action_selected_pair[index, target_id])
            pair_name = f"action_{target_id}_future_pair.png"
            _pair_route_image(
                image, tracks[index], visibility[index], future[index], pair_index,
                f"action {target_id}: selected future interaction pair",
            ).save(case_dir / pair_name)
            route_files.append(pair_name)
            for route_name, route_attention in (
                ("semantic", action_semantic_attention),
                ("motion", action_motion_attention),
            ):
                route_track = int(route_attention[index, target_id].argmax())
                route_file = f"action_{target_id}_{route_name}_route.png"
                _route_image(
                    image, tracks[index], visibility[index], future[index],
                    route_track, int(action_control[index, target_id]),
                    f"action {target_id} {route_name} route; mix={float(action_motion_mix[index, target_id]):.3f}",
                ).save(case_dir / route_file)
                route_files.append(route_file)
        top_reasons = reason_delta[index].abs().topk(min(3, reason_delta.shape[1])).indices
        for target_tensor in top_reasons:
            target_id = int(target_tensor)
            name = f"reason_{target_id}_object_intent.png"
            _route_image(
                image, tracks[index], visibility[index], future[index],
                int(reason_selected[index, target_id]), int(reason_control[index, target_id]),
                f"reason {target_id}: utility={float(reason_utility[index, target_id]):.3f} "
                f"selected={bool(reason_utility_selected[index, target_id])} "
                f"scale={float(reason_deploy_scale[index, target_id]):.1f}",
            ).save(case_dir / name)
            route_files.append(name)
            pair_index = int(reason_selected_pair[index, target_id])
            pair_name = f"reason_{target_id}_future_pair.png"
            _pair_route_image(
                image, tracks[index], visibility[index], future[index], pair_index,
                f"reason {target_id}: selected future interaction pair",
            ).save(case_dir / pair_name)
            route_files.append(pair_name)
            for route_name, route_attention in (
                ("semantic", reason_semantic_attention),
                ("motion", reason_motion_attention),
            ):
                route_track = int(route_attention[index, target_id].argmax())
                route_file = f"reason_{target_id}_{route_name}_route.png"
                _route_image(
                    image, tracks[index], visibility[index], future[index],
                    route_track, int(reason_control[index, target_id]),
                    f"reason {target_id} {route_name} route; mix={float(reason_motion_mix[index, target_id]):.3f}",
                ).save(case_dir / route_file)
                route_files.append(route_file)
        action_sign = 2 * action_target[index] - 1
        reason_sign = 2 * reason_target[index] - 1
        contribution = {
            "file_name": file_name,
            "interaction_risk_by_track": risk[index].tolist(),
            "future_ego_distance_by_track_and_horizon": future_ego_distance[index].tolist(),
            "future_approach_risk_by_track": future_approach_risk[index].tolist(),
            "action": {
                "utility": action_utility[index].tolist(),
                "utility_selected": action_utility_selected[index].tolist(),
                "utility_cutoff": action_utility_cutoff[index].tolist(),
                "deploy_scale": action_deploy_scale[index].tolist(),
                "delta": action_delta[index].tolist(),
                "signed_margin": (action_sign * action_delta[index]).tolist(),
                "selected_minus_control_deletion": (
                    action_sign * (action_control_deleted[index] - action_selected_deleted[index])
                ).tolist(),
                "selected_track": action_selected[index].tolist(),
                "control_track": action_control[index].tolist(),
                "attention": action_attention[index].tolist(),
                "semantic_attention": action_semantic_attention[index].tolist(),
                "motion_attention": action_motion_attention[index].tolist(),
                "motion_mix": action_motion_mix[index].tolist(),
                "pair_candidate": action_pair_candidate[index].tolist(),
                "selected_pair": action_selected_pair[index].tolist(),
                "control_pair": action_control_pair[index].tolist(),
                "pair_attention": action_pair_attention[index].tolist(),
            },
            "reason": {
                "utility": reason_utility[index].tolist(),
                "utility_selected": reason_utility_selected[index].tolist(),
                "utility_cutoff": reason_utility_cutoff[index].tolist(),
                "deploy_scale": reason_deploy_scale[index].tolist(),
                "delta": reason_delta[index].tolist(),
                "signed_margin": (reason_sign * reason_delta[index]).tolist(),
                "selected_minus_control_deletion": (
                    reason_sign * (reason_control_deleted[index] - reason_selected_deleted[index])
                ).tolist(),
                "selected_track": reason_selected[index].tolist(),
                "control_track": reason_control[index].tolist(),
                "attention": reason_attention[index].tolist(),
                "semantic_attention": reason_semantic_attention[index].tolist(),
                "motion_attention": reason_motion_attention[index].tolist(),
                "motion_mix": reason_motion_mix[index].tolist(),
                "pair_candidate": reason_pair_candidate[index].tolist(),
                "selected_pair": reason_selected_pair[index].tolist(),
                "control_pair": reason_control_pair[index].tolist(),
                "pair_attention": reason_pair_attention[index].tolist(),
            },
            "pair_geometry": {
                "min_future_distance": pair_min_future_distance[index].tolist(),
                "distance_reduction": pair_distance_reduction[index].tolist(),
            },
        }
        (case_dir / "object_intent_contributions.json").write_text(
            json.dumps(contribution, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (case_dir / "future_interaction_pairs.json").write_text(
            json.dumps(contribution["pair_geometry"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (case_dir / "report.html").write_text(
            "<html><body><h1>Object-intent traffic evidence</h1>"
            "<p>Red is target-selected traffic; gray is a support-matched control; "
            "yellow is the ego-compensated multi-horizon future.</p>"
            '<img src="original.jpg" style="max-width:100%">'
            + "".join(f'<img src="{name}" style="max-width:48%;margin:4px">' for name in route_files)
            + f"<pre>{html.escape(json.dumps(contribution, ensure_ascii=False, indent=2))}</pre>"
            "</body></html>", encoding="utf-8",
        )
        links.append(f'<li><a href="{case_dir.name}/report.html">{html.escape(file_name)}</a></li>')
    (output_dir / "index.html").write_text(
        "<html><body><h1>TIDA object-intent cases</h1><ul>" + "".join(links)
        + "</ul></body></html>", encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=32)
    args = parser.parse_args()
    export_object_intent_visuals(
        Path(args.epoch_dir), Path(args.manifest), Path(args.output_dir), max_cases=args.max_cases
    )


if __name__ == "__main__":
    main()
