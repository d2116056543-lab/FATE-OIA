from __future__ import annotations

from pathlib import Path
from typing import Any

from fate_oia.datasets.bdd100k_grounding import BDD100KGroundingIndex, bdd_oia_base_stem, load_bdd100k_objects


def build_scene_state_proxy(file_name: str, bdd100k_root: str | Path) -> dict[str, Any]:
    index = BDD100KGroundingIndex(bdd100k_root)
    paths = index.lookup(file_name)
    states = {
        "traffic_control_present": 0.0,
        "front_vehicle_or_obstacle": 0.0,
        "vulnerable_user_front": 0.0,
        "left_lane_structure": 0.0,
        "right_lane_structure": 0.0,
        "lower_center_drivable": 0.0,
    }
    available = False
    if paths.label_json:
        available = True
        for obj in load_bdd100k_objects(paths.label_json):
            cat = str(obj.get("category", "")).lower()
            box = obj.get("box2d") or {}
            x1, x2 = float(box.get("x1", 0)), float(box.get("x2", 0))
            y1, y2 = float(box.get("y1", 0)), float(box.get("y2", 0))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if "traffic light" in cat or "traffic sign" in cat:
                states["traffic_control_present"] = 1.0
            if any(k in cat for k in ["car", "truck", "bus", "motor"]) and 450 <= cx <= 830 and cy >= 250:
                states["front_vehicle_or_obstacle"] = 1.0
            if any(k in cat for k in ["person", "rider", "bike"]) and 420 <= cx <= 860 and cy >= 220:
                states["vulnerable_user_front"] = 1.0
            if cx < 576:
                states["left_lane_structure"] = max(states["left_lane_structure"], 0.5)
            if cx > 704:
                states["right_lane_structure"] = max(states["right_lane_structure"], 0.5)
    if paths.drivable_map:
        available = True
        states["lower_center_drivable"] = 1.0
    return {"file_name": file_name, "base_stem": bdd_oia_base_stem(file_name), "states": states, "available": available, "weak_proxy": True, "semantic_segmentation_full_split": False}
