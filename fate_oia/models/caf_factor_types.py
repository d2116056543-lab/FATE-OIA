from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class FactorGroup(IntEnum):
    ACTOR_EVIDENCE = 0
    DINO_OBJECT_AFFINITY = 1
    SCENE_STATE_WEAK = 2
    LANE_DRIVABLE = 3
    TRAFFIC_CONTROL = 4
    GLOBAL_CONTEXT = 5
    OTHER = 6


@dataclass(frozen=True)
class FactorMeta:
    group: int
    source: str
    name: str
    weak: bool = False


DEFAULT_FACTOR_GROUP_NAMES = {
    int(FactorGroup.ACTOR_EVIDENCE): "actor_evidence",
    int(FactorGroup.DINO_OBJECT_AFFINITY): "dino_object_affinity",
    int(FactorGroup.SCENE_STATE_WEAK): "bdd100k_scene_state_weak",
    int(FactorGroup.LANE_DRIVABLE): "lane_drivable",
    int(FactorGroup.TRAFFIC_CONTROL): "traffic_control",
    int(FactorGroup.GLOBAL_CONTEXT): "global_context",
    int(FactorGroup.OTHER): "other",
}
