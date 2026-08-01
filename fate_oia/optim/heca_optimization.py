from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor


class HECALossRegistry:
    """Single source of truth that prevents duplicate weighted loss terms."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}

    def add(self, name: str, value: Tensor, weight: float, *, owner: str) -> None:
        if name in self._rows:
            raise ValueError(f"HECA loss term added twice: {name}")
        if value.ndim != 0:
            raise ValueError(f"HECA loss term must be scalar: {name}")
        self._rows[name] = {
            "value": value,
            "weight": float(weight),
            "owner": str(owner),
        }

    def total(self) -> Tensor:
        if not self._rows:
            return torch.zeros(())
        return sum(row["weight"] * row["value"] for row in self._rows.values())

    def owner_total(self, owners: set[str]) -> Tensor:
        rows = [
            row["weight"] * row["value"]
            for row in self._rows.values()
            if row["owner"] in owners
        ]
        if not rows:
            return self.total().new_zeros(())
        return sum(rows)

    def artifact(self) -> list[dict[str, Any]]:
        return [
            {
                "term": name,
                "owner": row["owner"],
                "weight": row["weight"],
                "call_count": 1,
                "value": float(row["value"].detach()),
                "weighted_value": float((row["weight"] * row["value"]).detach()),
            }
            for name, row in self._rows.items()
        ]


class HECAExcessRiskBalancer:
    def __init__(self, *, momentum: float = 0.99, temperature: float = 1.0) -> None:
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.action_ema: float | None = None
        self.reason_ema: float | None = None
        self.action_floor: float | None = None
        self.reason_floor: float | None = None

    def update_floors(self, action_loss: Tensor, reason_loss: Tensor) -> None:
        values = (float(action_loss.detach()), float(reason_loss.detach()))
        if not all(torch.isfinite(torch.tensor(values))):
            raise ValueError("HECA excess-risk floors require finite losses")
        if self.action_ema is None:
            self.action_ema, self.reason_ema = values
            self.action_floor, self.reason_floor = values
            return
        self.action_ema = self.momentum * self.action_ema + (1 - self.momentum) * values[0]
        self.reason_ema = self.momentum * self.reason_ema + (1 - self.momentum) * values[1]
        self.action_floor = min(float(self.action_floor), self.action_ema)
        self.reason_floor = min(float(self.reason_floor), self.reason_ema)

    def weights(self, action_loss: Tensor, reason_loss: Tensor) -> dict[str, float]:
        if self.action_floor is None or self.reason_floor is None:
            self.update_floors(action_loss, reason_loss)
        excess = torch.tensor(
            [
                (float(action_loss.detach()) - float(self.action_floor))
                / (abs(float(self.action_floor)) + 1e-6),
                (float(reason_loss.detach()) - float(self.reason_floor))
                / (abs(float(self.reason_floor)) + 1e-6),
            ]
        )
        raw = torch.softmax(excess / self.temperature, dim=0)
        action = min(max(float(raw[0]), 0.45), 0.70)
        return {"action": action, "reason": 1.0 - action}

    def state_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class ReasonProbabilityEMA:
    """Per-training-sample reason probability EMA; never reads test labels."""

    def __init__(self, momentum: float = 0.90) -> None:
        self.momentum = float(momentum)
        self.values: dict[str, Tensor] = {}

    def values_for(self, file_names: list[str], fallback: Tensor) -> Tensor:
        rows = [self.values.get(str(name), fallback[index].detach().cpu()) for index, name in enumerate(file_names)]
        return torch.stack(rows).to(device=fallback.device, dtype=fallback.dtype)

    def update(self, file_names: list[str], probability: Tensor) -> None:
        for name, row in zip(file_names, probability.detach().float().cpu()):
            key = str(name)
            previous = self.values.get(key)
            self.values[key] = row.clone() if previous is None else self.momentum * previous + (1.0 - self.momentum) * row

    def state_dict(self) -> dict[str, Any]:
        return {"momentum": self.momentum, "values": self.values}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.momentum = float(state.get("momentum", self.momentum))
        self.values = {str(key): value.detach().cpu() for key, value in dict(state.get("values", {})).items()}


def identity_corruption_mode(microbatch_index: int) -> str:
    return ("schema", "cross_sample", "state")[int(microbatch_index) % 3]


def correction_fraction_for_run(run_kind: str, *, gate_c_pass: bool) -> float:
    if run_kind == "pilot":
        return 0.20
    if run_kind == "full" and gate_c_pass:
        return 0.25
    if run_kind == "full":
        raise ValueError("HECA full correction fraction requires Gate C")
    raise ValueError(f"Unknown HECA run kind: {run_kind}")


def validate_formal_protocol(config: dict[str, Any]) -> None:
    if config.get("from_scratch") is not True:
        raise ValueError("Formal HECA run must start from scratch")
    if int(config.get("epochs", -1)) != 14:
        raise ValueError("Formal HECA run requires exactly 14 epochs")
    if config.get("pilot_checkpoint") not in (None, ""):
        raise ValueError("Formal HECA run cannot resume a pilot checkpoint")


@dataclass
class HECAScheduleState:
    update: int
    total_updates: int
    corruption_phase: int = 0
    corruption_microbatch_index: int = 0
    foundation_grad_ema: float = 0.0
    action_floor: float | None = None
    reason_floor: float | None = None
    visual_rms_ema: list[float] = field(default_factory=lambda: [1.0] * 4)
    pu_pass_streak: list[int] = field(default_factory=lambda: [0] * 21)
    foundation_lr_hold: bool = False

    def progress(self) -> float:
        return self.update / max(self.total_updates, 1)

    def foundation_lr_multiplier(self, *, logit_rms: float) -> float:
        progress = self.progress()
        if progress <= 0.05:
            return max(progress / 0.05, 0.0) * 0.5
        allowed = logit_rms < 8.0 and self.foundation_grad_ema < 5.0
        if not allowed:
            self.foundation_lr_hold = True
            return 0.5
        if progress <= 0.20:
            return 0.5 + 0.5 * (progress - 0.05) / 0.15
        return 1.0

    def foundation_grad_cap(self, minimum: float, maximum: float) -> float:
        pressure = min(max(self.foundation_grad_ema / 5.0, 0.0), 1.0)
        return float(maximum) - (float(maximum) - float(minimum)) * pressure

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "HECAScheduleState":
        return cls(**state)
