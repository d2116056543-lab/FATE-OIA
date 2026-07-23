from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


_SCALE_FIELDS = (
    "foundation",
    "evidence",
    "reread",
    "annotation",
    "exchange",
    "reason_latent",
    "intervention",
    "threshold",
)


@dataclass(frozen=True)
class PRECISECurriculumState:
    epoch: int
    stage: str
    foundation: float
    evidence: float
    reread: float
    annotation: float
    exchange: float
    reason_latent: float
    intervention: float
    threshold: float

    @property
    def owner_active(self) -> dict[str, bool]:
        return {
            "action_foundation": self.foundation > 0.0,
            "action_decoder": self.foundation > 0.0,
            "reason_semantic": self.foundation > 0.0,
            "evidence_core": self.evidence > 0.0,
            "reread_adapter": self.reread > 0.0,
            "annotation_adapter": self.annotation > 0.0,
            "threshold_head": self.threshold > 0.0,
            "exchange_adapter": self.exchange > 0.0,
            "reason_latent": self.reason_latent > 0.0,
        }

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["owner_active"] = self.owner_active
        return value


def _curriculum(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("curriculum")
    if not isinstance(value, dict):
        raise ValueError("config is missing the PRECISE curriculum")
    if int(value.get("epochs", -1)) != 12:
        raise ValueError("PRECISE full curriculum must define exactly 12 epochs")
    schedule = value.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("PRECISE curriculum schedule is empty")
    return value


def curriculum_state_for_epoch(config: dict[str, Any], epoch: int) -> PRECISECurriculumState:
    curriculum = _curriculum(config)
    if epoch < 0 or epoch >= int(curriculum["epochs"]):
        raise ValueError(f"epoch {epoch} is outside the formal PRECISE run")
    rows = [row for row in curriculum["schedule"] if int(row["start_epoch"]) <= epoch <= int(row["end_epoch"])]
    if len(rows) != 1:
        raise ValueError(f"epoch {epoch} must match exactly one curriculum row")
    row = rows[0]
    scales = {name: float(row[name]) for name in _SCALE_FIELDS}
    if any(value < 0.0 or value > 1.0 for value in scales.values()):
        raise ValueError(f"epoch {epoch} has an activation outside [0, 1]")
    return PRECISECurriculumState(epoch=epoch, stage=str(row["stage"]), **scales)


def owner_active_epoch_counts(config: dict[str, Any], epochs: int) -> dict[str, int]:
    if epochs != int(_curriculum(config)["epochs"]):
        raise ValueError("runtime epochs must equal the fixed curriculum length")
    counts: dict[str, int] = {}
    for epoch in range(epochs):
        for owner, active in curriculum_state_for_epoch(config, epoch).owner_active.items():
            counts[owner] = counts.get(owner, 0) + int(active)
    return counts


def curriculum_sha256(config: dict[str, Any]) -> str:
    normalized = json.dumps(_curriculum(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()
