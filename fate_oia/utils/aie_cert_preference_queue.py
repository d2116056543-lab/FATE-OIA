from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PreferenceBatch:
    primary_reason_logits: Tensor
    final_reason_logits: Tensor
    reason_target: Tensor
    verified_counter: Tensor
    counter_reliability: Tensor
    enqueue_update: Tensor
    sample_id: list[str]


class AIECertPreferenceQueue:
    def __init__(self, capacity: int = 512, max_age: int = 64, age_tau: float = 32.0) -> None:
        self.capacity = int(capacity)
        self.max_age = int(max_age)
        self.age_tau = float(age_tau)
        self.records: list[dict] = []

    def enqueue(self, batch: PreferenceBatch) -> None:
        for index, sample_id in enumerate(batch.sample_id):
            self.records.append({
                "primary_reason_logits": batch.primary_reason_logits[index].detach().cpu(),
                "final_reason_logits": batch.final_reason_logits[index].detach().cpu(),
                "reason_target": batch.reason_target[index].detach().cpu(),
                "verified_counter": batch.verified_counter[index].detach().cpu(),
                "counter_reliability": batch.counter_reliability[index].detach().cpu(),
                "enqueue_update": int(batch.enqueue_update[index]),
                "sample_id": sample_id,
            })
        self.records = self.records[-self.capacity :]

    def eligible(self, update: int) -> list[dict]:
        return [row for row in self.records if 0 <= update - row["enqueue_update"] <= self.max_age]

    def state_dict(self) -> dict:
        return {"records": self.records, "capacity": self.capacity, "max_age": self.max_age, "age_tau": self.age_tau}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self.max_age = int(state["max_age"])
        self.age_tau = float(state["age_tau"])
        self.records = list(state["records"])[-self.capacity :]
