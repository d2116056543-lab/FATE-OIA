from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROLE_BY_CATEGORY = {
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "train": "vehicle",
    "motor": "vulnerable_road_user",
    "bike": "vulnerable_road_user",
    "rider": "vulnerable_road_user",
    "person": "vulnerable_road_user",
    "traffic light": "traffic_control",
    "traffic sign": "traffic_control",
}


def _read_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["partition"] == "train_core":
                rows.append(row)
                if len(rows) == limit:
                    break
    return rows


def _image_mse(path_a: Path, path_b: Path) -> float:
    with Image.open(path_a) as image_a, Image.open(path_b) as image_b:
        a = np.asarray(image_a.convert("RGB").resize((160, 90)), dtype=np.float32) / 255.0
        b = np.asarray(image_b.convert("RGB").resize((160, 90)), dtype=np.float32) / 255.0
    return float(np.square(a - b).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--track_store", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, required=True)
    parser.add_argument("--label_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = _read_rows(args.manifest, args.limit)
    payload = torch.load(args.track_store, map_location="cpu", weights_only=True)
    tracks = {
        str(name).lower(): (payload["tracks_xy"][index].float(), payload["visibility"][index].bool())
        for index, name in enumerate(payload["file_names"])
    }
    categories: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    point_roles: Counter[str] = Counter()
    image_mse = []
    annotation_count = 0
    visible_points = 0
    boxed_points = 0
    missing = []

    for row in rows:
        video_id = row["source_video_id"]
        official_split = row["official_split"]
        image_path = args.image_root / official_split / f"{video_id}.jpg"
        label_path = args.label_root / official_split / f"{video_id}.json"
        target_path = Path(row["target_image_path"])
        if not image_path.is_file() or not label_path.is_file() or not target_path.is_file():
            missing.append(row["file_name"])
            continue
        image_mse.append(_image_mse(target_path, image_path))
        annotation = json.loads(label_path.read_text(encoding="utf-8"))
        objects = annotation["frames"][-1].get("objects", [])
        boxes = []
        for obj in objects:
            category = str(obj.get("category", "unknown"))
            role = ROLE_BY_CATEGORY.get(category)
            categories[category] += 1
            if role is None or "box2d" not in obj:
                continue
            roles[role] += 1
            box = obj["box2d"]
            boxes.append((role, box["x1"], box["y1"], box["x2"], box["y2"]))
        annotation_count += 1

        key = str(row["file_name"]).lower()
        if key not in tracks:
            continue
        xy, visibility = tracks[key]
        terminal_xy = xy[-1]
        terminal_visible = visibility[-1]
        for point, is_visible in zip(terminal_xy, terminal_visible):
            if not bool(is_visible):
                continue
            visible_points += 1
            x = float((point[0].clamp(-1, 1) + 1) * 640.0)
            y = float((point[1].clamp(-1, 1) + 1) * 360.0)
            matched = [box for box in boxes if box[1] <= x <= box[3] and box[2] <= y <= box[4]]
            if matched:
                matched.sort(key=lambda box: (box[3] - box[1]) * (box[4] - box[2]))
                point_roles[matched[0][0]] += 1
                boxed_points += 1
            else:
                point_roles["background"] += 1

    mse_array = np.asarray(image_mse, dtype=np.float64)
    result = {
        "requested_rows": len(rows),
        "audited_rows": annotation_count,
        "missing_rows": len(missing),
        "image_mse_mean": float(mse_array.mean()) if mse_array.size else None,
        "image_mse_p50": float(np.quantile(mse_array, 0.50)) if mse_array.size else None,
        "image_mse_p95": float(np.quantile(mse_array, 0.95)) if mse_array.size else None,
        "image_mse_lt_0p01_rate": float((mse_array < 0.01).mean()) if mse_array.size else None,
        "object_categories": dict(categories),
        "object_roles": dict(roles),
        "visible_terminal_points": visible_points,
        "boxed_terminal_points": boxed_points,
        "boxed_terminal_point_rate": boxed_points / max(visible_points, 1),
        "terminal_point_roles": dict(point_roles),
        "missing_examples": missing[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
