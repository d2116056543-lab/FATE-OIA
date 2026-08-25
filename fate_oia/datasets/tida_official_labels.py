from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ActionMap = dict[str, tuple[float, float, float, float]]
ReasonMap = dict[str, tuple[float, ...]]
OfficialLabelMap = dict[str, tuple[tuple[float, ...], tuple[float, ...]]]


def _name(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("label row has no file name")
    return Path(value.replace("\\", "/")).name.lower()


def _insert_unique(target: dict[str, Any], name: str, value: Any) -> None:
    if name in target:
        raise ValueError(f"duplicate label file name: {name}")
    target[name] = value


def load_action_label_map(path: str | Path) -> ActionMap:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("action labels must use the BDD-OIA COCO schema")
    images: dict[int, str] = {}
    names: set[str] = set()
    for row in payload["images"]:
        image_id = int(row["id"])
        name = _name(row["file_name"])
        if image_id in images or name in names:
            raise ValueError(f"duplicate action image id or file name: {image_id}/{name}")
        images[image_id] = name
        names.add(name)
    result: ActionMap = {}
    for row in payload.get("annotations", []):
        image_id = int(row.get("image_id", row.get("id")))
        if image_id not in images:
            raise ValueError(f"action annotation has no image: {image_id}")
        values = row.get("category")
        if not isinstance(values, list) or len(values) < 4:
            raise ValueError("action annotation requires at least 4 values")
        _insert_unique(result, images[image_id], tuple(float(value) for value in values[:4]))
    if set(result) != set(images.values()):
        missing = sorted(set(images.values()) - set(result))
        raise ValueError(f"action images missing annotations: {missing[:5]}")
    return result


def load_reason_label_map(path: str | Path) -> ReasonMap:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("reason labels must use the BDD-OIA list schema")
    result: ReasonMap = {}
    for row in payload:
        name = _name(row.get("file_name"))
        values = row.get("reason")
        if not isinstance(values, list) or len(values) != 21:
            raise ValueError("reason annotation requires exactly 21 values")
        _insert_unique(result, name, tuple(float(value) for value in values))
    return result


def load_official_label_map(action_path: str | Path, reason_path: str | Path) -> OfficialLabelMap:
    actions = load_action_label_map(action_path)
    reasons = load_reason_label_map(reason_path)
    if set(actions) != set(reasons):
        action_only = sorted(set(actions) - set(reasons))[:5]
        reason_only = sorted(set(reasons) - set(actions))[:5]
        raise ValueError(f"label key mismatch: action_only={action_only}, reason_only={reason_only}")
    return {name: (actions[name], reasons[name]) for name in sorted(actions)}
