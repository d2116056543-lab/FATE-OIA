from __future__ import annotations

from typing import Any, Sequence

import torch


def translate_dynamic_concepts(
    predicate_names: Sequence[str],
    region_mass_velocity: torch.Tensor,
    reliability: torch.Tensor,
    reliability_threshold: float = 0.25,
    motion_threshold: float = 0.01,
) -> list[dict[str, Any]]:
    velocity = region_mass_velocity.detach().float().cpu()
    rho = reliability.detach().float().cpu()
    results: list[dict[str, Any]] = []
    for batch in range(velocity.shape[0]):
        concepts: dict[str, Any] = {}
        for index, name in enumerate(predicate_names):
            confidence = float(rho[batch, index])
            left, right, front, upper, _global = (float(value) for value in velocity[batch, index])
            if confidence < reliability_threshold:
                concept = "unknown"
            elif name in ("front_vehicle_close", "front_vehicle_far") and abs(front) >= motion_threshold:
                concept = "front_closing" if front > 0 else "front_receding"
            elif name == "road_crowded" and abs(front) >= motion_threshold:
                concept = "queue_building" if front > 0 else "queue_dissipating"
            elif name in ("vehicle_left", "parked_vehicle_left") and abs(left) >= motion_threshold:
                concept = "left_inflow" if left > 0 else "left_gap_opening"
            elif name == "open_left_gap" and abs(left) >= motion_threshold:
                concept = "left_gap_opening" if left < 0 else "left_inflow"
            elif name in ("vehicle_right", "parked_vehicle_right") and abs(right) >= motion_threshold:
                concept = "right_inflow" if right > 0 else "right_gap_opening"
            elif name == "open_right_gap" and abs(right) >= motion_threshold:
                concept = "right_gap_opening" if right < 0 else "right_inflow"
            elif name in ("pedestrian_front", "cyclist_front") and abs(right - left) >= motion_threshold:
                concept = "crossing_left_to_right" if right > left else "crossing_right_to_left"
            elif name == "traffic_light_red":
                concept = "signal_turning_red" if upper >= motion_threshold else "unknown"
            elif name == "traffic_light_green":
                concept = "signal_turning_green" if upper >= motion_threshold else "unknown"
            else:
                concept = "dynamic_change" if max(abs(left), abs(right), abs(front), abs(upper)) >= motion_threshold else "unknown"
            concepts[name] = concept
            concepts[f"{name}_confidence"] = confidence
            concepts[f"{name}_region_velocity"] = [left, right, front, upper]
        results.append(concepts)
    return results
