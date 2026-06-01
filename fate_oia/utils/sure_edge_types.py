from __future__ import annotations

EDGE_TYPES = [
    "object_box",
    "lane_poly",
    "drivable_area",
    "scene_attribute",
    "patch_context",
]

EDGE_TYPE_TO_ID = {name: idx for idx, name in enumerate(EDGE_TYPES)}


def edge_type_id(name: str) -> int:
    return EDGE_TYPE_TO_ID.get(name, EDGE_TYPE_TO_ID["patch_context"])
