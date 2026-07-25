"""RAEL task-aware grounding targets and exact Hungarian entity matching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord


def _category(value: Mapping[str, Any]) -> str:
    return str(value.get("category") or value.get("type") or "unknown").replace(" ", "_").lower()


def _box(value: Mapping[str, Any]) -> tuple[float, float, float, float]:
    raw = value["box"]
    if len(raw) != 4:
        raise ValueError("entity box must have [x1, y1, x2, y2]")
    box = tuple(float(item) for item in raw)
    if not all(math.isfinite(item) for item in box) or box[0] >= box[2] or box[1] >= box[3]:
        raise ValueError("entity box must be finite with x1 < x2 and y1 < y2")
    return box  # type: ignore[return-value]


def _sector(value: Mapping[str, Any]) -> str:
    return str(value.get("sector") or value.get("side") or "unknown").lower()


def _giou_loss(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0.0 else 0.0
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    enclosure = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    giou = iou - ((enclosure - union) / enclosure if enclosure > 0.0 else 0.0)
    return 1.0 - giou


def entity_match_cost(
    slot: Mapping[str, Any],
    detection: Mapping[str, Any],
    *,
    image_size: tuple[int, int],
    return_components: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Use the plan's exact 1/2/2/0.5 type, L1, GIoU, sector contract."""

    width, height = image_size
    a, b = _box(slot), _box(detection)
    norm = (max(float(width), 1.0), max(float(height), 1.0))
    box_l1 = sum(abs(x - y) / norm[index % 2] for index, (x, y) in enumerate(zip(a, b))) / 4.0
    components = {
        "type": 0.0 if _category(slot) == _category(detection) else 1.0,
        "box_l1": box_l1,
        "giou": _giou_loss(a, b),
        "sector": 0.0 if _sector(slot) == _sector(detection) else 1.0,
    }
    cost = 1.0 * components["type"] + 2.0 * components["box_l1"] + 2.0 * components["giou"] + 0.5 * components["sector"]
    return (cost, components) if return_components else cost


def _hungarian_minimise(cost: list[list[float]]) -> list[tuple[int, int]]:
    """Pure-Python Hungarian solver; avoids a SciPy runtime dependency."""

    if not cost or not cost[0]:
        return []
    transposed = len(cost) > len(cost[0])
    matrix = [list(row) for row in zip(*cost)] if transposed else cost
    rows, cols = len(matrix), len(matrix[0])
    u, v = [0.0] * (rows + 1), [0.0] * (cols + 1)
    p, way = [0] * (cols + 1), [0] * (cols + 1)
    for row in range(1, rows + 1):
        p[0], col0 = row, 0
        minv, used = [float("inf")] * (cols + 1), [False] * (cols + 1)
        while True:
            used[col0] = True
            row0, delta, next_col = p[col0], float("inf"), 0
            for col in range(1, cols + 1):
                if used[col]:
                    continue
                cur = matrix[row0 - 1][col - 1] - u[row0] - v[col]
                if cur < minv[col]:
                    minv[col], way[col] = cur, col0
                if minv[col] < delta:
                    delta, next_col = minv[col], col
            for col in range(cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    minv[col] -= delta
            col0 = next_col
            if p[col0] == 0:
                break
        while True:
            previous = way[col0]
            p[col0] = p[previous]
            col0 = previous
            if col0 == 0:
                break
    pairs = [(p[col] - 1, col - 1) for col in range(1, cols + 1) if p[col] != 0]
    return [(col, row) for row, col in pairs] if transposed else pairs


@dataclass(frozen=True)
class EntityAssignment:
    slot_index: int
    detection_index: int
    cost: float


def hungarian_entity_matching(
    slots: Iterable[Mapping[str, Any]],
    detections: Iterable[Mapping[str, Any]],
    *,
    image_size: tuple[int, int],
) -> tuple[EntityAssignment, ...]:
    slot_list, detection_list = list(slots), list(detections)
    matrix = [
        [float(entity_match_cost(slot, detection, image_size=image_size)) for detection in detection_list]
        for slot in slot_list
    ]
    return tuple(
        EntityAssignment(slot_index, detection_index, matrix[slot_index][detection_index])
        for slot_index, detection_index in _hungarian_minimise(matrix)
    )


@dataclass(frozen=True)
class ObjectnessTarget:
    target: float
    reliable: bool
    matched_detection_index: int | None


@dataclass(frozen=True)
class TrafficStateTarget:
    detection_index: int
    matched_slot_index: int | None
    state: str | None
    valid: bool


@dataclass(frozen=True)
class EntityGroundingTargets:
    assignments: tuple[EntityAssignment, ...]
    objectness: tuple[ObjectnessTarget, ...]
    traffic_state_targets: tuple[TrafficStateTarget, ...]
    coverage: dict[str, int]


def _coverage_base() -> dict[str, int]:
    return {
        "matched_entity_count": 0,
        "unmatched_positive_count": 0,
        "reliable_negative_count": 0,
        "unknown_count": 0,
        "traffic_state_valid_count": 0,
        "drivable_valid_count": 0,
        "boundary_valid_count": 0,
    }


def _traffic_state(attributes: Mapping[str, Any]) -> str | None:
    """Return only explicit, usable traffic-light attributes; unknown stays unknown."""

    if "trafficLightColor" in attributes:
        raw = attributes["trafficLightColor"]
    elif "traffic_light_color" in attributes:
        raw = attributes["traffic_light_color"]
    else:
        return None
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "red": "red",
        "green": "green",
        "yellow": "yellow_or_other",
        "amber": "yellow_or_other",
        "yellow_or_other": "yellow_or_other",
        "other": "yellow_or_other",
    }.get(value)


def build_entity_grounding_targets(
    slots: Iterable[Mapping[str, Any]],
    record: RAELGroundingRecord,
    *,
    image_size: tuple[int, int],
) -> EntityGroundingTargets:
    slot_list, detections = list(slots), list(record.detections)
    assignments = hungarian_entity_matching(slot_list, detections, image_size=image_size)
    by_slot = {assignment.slot_index: assignment for assignment in assignments}
    matched_detections = {assignment.detection_index for assignment in assignments}
    complete = bool(record.source_complete.get("detections", False))
    coverage = _coverage_base()
    coverage["matched_entity_count"] = len(assignments)
    coverage["unmatched_positive_count"] = max(0, len(detections) - len(matched_detections))
    objectness: list[ObjectnessTarget] = []
    for index, _slot in enumerate(slot_list):
        assignment = by_slot.get(index)
        if assignment is not None:
            objectness.append(ObjectnessTarget(1.0, True, assignment.detection_index))
        elif complete:
            objectness.append(ObjectnessTarget(0.0, True, None))
            coverage["reliable_negative_count"] += 1
        else:
            objectness.append(ObjectnessTarget(0.0, False, None))
            coverage["unknown_count"] += 1
    slot_by_detection = {assignment.detection_index: assignment.slot_index for assignment in assignments}
    traffic: list[TrafficStateTarget] = []
    for detection_index, detection in enumerate(detections):
        attributes = detection.get("attributes") or {}
        if _category(detection) in {"traffic_light", "traffic_control"}:
            state = _traffic_state(attributes if isinstance(attributes, Mapping) else {})
            matched_slot_index = slot_by_detection.get(detection_index)
            valid = state is not None and matched_slot_index is not None
            traffic.append(TrafficStateTarget(detection_index, matched_slot_index, state, valid))
    coverage["traffic_state_valid_count"] = sum(target.valid for target in traffic)
    return EntityGroundingTargets(assignments, tuple(objectness), tuple(traffic), coverage)


@dataclass(frozen=True)
class RoadGroundingTargets:
    drivable: tuple[dict[str, Any], ...]
    boundaries: tuple[dict[str, Any], ...]
    drivable_valid_mask: tuple[bool, ...]
    boundary_valid_mask: tuple[bool, ...]
    active_boundary_loss: bool
    coverage: dict[str, int]


def _coordinates(value: Any) -> list[tuple[float, float]]:
    if isinstance(value, Mapping):
        if isinstance(value.get("x"), (int, float)) and isinstance(value.get("y"), (int, float)):
            return [(float(value["x"]), float(value["y"]))]
        points: list[tuple[float, float]] = []
        for item in value.values():
            points.extend(_coordinates(item))
        return points
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            return [(float(value[0]), float(value[1]))]
        points = []
        for item in value:
            points.extend(_coordinates(item))
        return points
    return []


def _road_sector(item: Mapping[str, Any], geometry_key: str, *, image_width: float) -> str | None:
    """Use side when declared, otherwise derive one sector from actual geometry."""

    declared = _sector(item)
    if declared in {"left", "center", "right"}:
        return declared
    coordinates = _coordinates(item.get(geometry_key))
    if not coordinates:
        return None
    mean_x = sum(x for x, _ in coordinates) / len(coordinates)
    width = 1.0 if max(abs(x) for x, _ in coordinates) <= 1.0 else max(image_width, 1.0)
    if mean_x < width / 3.0:
        return "left"
    if mean_x > (2.0 * width / 3.0):
        return "right"
    return "center"


def build_road_grounding_targets(
    record: RAELGroundingRecord,
    *,
    image_size: tuple[int, int] = (640, 360),
) -> RoadGroundingTargets:
    coverage = _coverage_base()
    drivable_mask = [False, False, False]
    boundary_mask = [False, False]
    if record.source_complete.get("drivable", False):
        for region in record.drivable:
            sector = _road_sector(region, "polygon", image_width=float(image_size[0]))
            if sector in {"left", "center", "right"}:
                drivable_mask[{"left": 0, "center": 1, "right": 2}[sector]] = True
    if record.source_complete.get("lanes", False):
        for boundary in record.lanes:
            sector = _road_sector(boundary, "points", image_width=float(image_size[0]))
            if sector in {"left", "right"}:
                boundary_mask[{"left": 0, "right": 1}[sector]] = True
    coverage["drivable_valid_count"] = sum(drivable_mask)
    coverage["boundary_valid_count"] = sum(boundary_mask)
    return RoadGroundingTargets(
        drivable=tuple(record.drivable),
        boundaries=tuple(record.lanes),
        drivable_valid_mask=tuple(drivable_mask),
        boundary_valid_mask=tuple(boundary_mask),
        active_boundary_loss=any(boundary_mask),
        coverage=coverage,
    )


def aggregate_grounding_coverage(
    entity_targets: Iterable[EntityGroundingTargets],
    road_targets: Iterable[RoadGroundingTargets],
) -> dict[str, int]:
    """Aggregate the seven P1 coverage fields without inventing labels."""

    coverage = _coverage_base()
    for target in entity_targets:
        for field in coverage:
            coverage[field] += int(target.coverage.get(field, 0))
    for target in road_targets:
        for field in ("drivable_valid_count", "boundary_valid_count"):
            coverage[field] += int(target.coverage.get(field, 0))
    return coverage
