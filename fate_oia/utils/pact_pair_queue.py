from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class PairBank:
    positive: list[tuple[Tensor, int]]
    negative: list[tuple[Tensor, int, float]]


class PACTBalancedPairQueue:
    """Detached per-label queue. Weak negatives keep every label trainable."""

    def __init__(self, labels: int, positive_capacity: int = 256, negative_capacity: int = 256,
                 max_age_updates: int = 64) -> None:
        self.labels = int(labels)
        self.positive_capacity = int(positive_capacity)
        self.negative_capacity = int(negative_capacity)
        self.max_age_updates = int(max_age_updates)
        self.banks = [PairBank([], []) for _ in range(self.labels)]

    def enqueue(self, logits: Tensor, targets: Tensor, update: int, counter_priority: Tensor | None = None) -> None:
        values, labels = logits.detach().cpu(), targets.detach().cpu()
        priority = torch.zeros_like(labels) if counter_priority is None else counter_priority.detach().cpu()
        for row in range(values.shape[0]):
            for label in range(self.labels):
                bank = self.banks[label]
                if labels[row, label] > .5:
                    bank.positive.append((values[row, label].clone(), int(update)))
                    del bank.positive[:-self.positive_capacity]
                else:
                    bank.negative.append((values[row, label].clone(), int(update), float(priority[row, label])))
                    bank.negative.sort(key=lambda item: item[2], reverse=True)
                    del bank.negative[self.negative_capacity:]

    def pairs(self, update: int, device: torch.device, cap_per_label: int = 8) -> tuple[list[tuple[int, Tensor, Tensor]], dict]:
        rows, positive_counts, negative_counts, pair_counts = [], [], [], []
        for label, bank in enumerate(self.banks):
            pos = [x for x in bank.positive if update - x[1] <= self.max_age_updates]
            neg = [x for x in bank.negative if update - x[1] <= self.max_age_updates]
            count = min(len(pos), len(neg), int(cap_per_label))
            if count:
                rows.append((label, torch.stack([x[0] for x in pos[-count:]]).to(device),
                             torch.stack([x[0] for x in neg[:count]]).to(device)))
            positive_counts.append(len(pos)); negative_counts.append(len(neg)); pair_counts.append(count)
        stats = {"positive_count": positive_counts, "negative_count": negative_counts,
                 "pair_count": pair_counts, "labels_with_pairs": sum(x > 0 for x in pair_counts)}
        return rows, stats

    def state_dict(self) -> dict:
        return {"labels": self.labels, "banks": self.banks}

    def load_state_dict(self, state: dict) -> None:
        if int(state["labels"]) != self.labels:
            raise ValueError("pair queue label count mismatch")
        restored = []
        for bank in state["banks"]:
            restored.append(PairBank(
                [(value.detach().cpu(), int(update)) for value, update in bank.positive],
                [(value.detach().cpu(), int(update), float(priority)) for value, update, priority in bank.negative],
            ))
        self.banks = restored
