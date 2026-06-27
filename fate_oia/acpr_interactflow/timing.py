from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


REQUIRED_TIMING_SECTIONS = [
    "data_gap",
    "h2d",
    "visual_dino",
    "visual_motion",
    "predicate",
    "interaction_flow",
    "response_lag",
    "decision_ledger",
    "exp29",
    "loss",
    "backward",
    "optimizer",
    "eval_forward",
    "artifact_write",
]


@dataclass
class StepTimer:
    """Small deterministic wall-clock timer for CALI-Flow++ profiling."""

    totals: dict[str, float] = field(default_factory=lambda: {k: 0.0 for k in REQUIRED_TIMING_SECTIONS})
    counts: dict[str, int] = field(default_factory=lambda: {k: 0 for k in REQUIRED_TIMING_SECTIONS})

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
        start = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1

    def add(self, name: str, seconds: float) -> None:
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
        self.totals[name] += float(seconds)
        self.counts[name] += 1

    def summary(self, reset: bool = True) -> dict[str, float]:
        total = sum(self.totals.values())
        row: dict[str, float] = {}
        for key in REQUIRED_TIMING_SECTIONS:
            value = float(self.totals.get(key, 0.0))
            row[f"{key}_time"] = value
            row[f"{key}_fraction"] = value / max(total, 1e-9)
        row["total_profiled_time"] = float(total)
        if reset:
            self.reset()
        return row

    def reset(self) -> None:
        for key in list(self.totals):
            self.totals[key] = 0.0
            self.counts[key] = 0
