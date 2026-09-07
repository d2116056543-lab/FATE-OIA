from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Sampler


def _metadata_sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CoEVBudgetedStratifiedSampler(Sampler[tuple[int, int]]):
    """Stateful rotating sampler with pixel-novelty strata and label coverage.

    Novelty decides the three strata without task labels. Labels are used only
    to choose a representative training subset inside each stratum.
    """

    def __init__(self, data_source: Sized, *, seed: int, metadata_path: str | Path,
                 epoch_size: int, quotas: dict[str, int], missing_history_quota: int = 0,
                 candidate_draws: int = 12) -> None:
        if not hasattr(data_source, "records"):
            raise TypeError("budget sampler requires a dataset exposing records")
        self.data_source = data_source
        self.seed = int(seed); self.epoch_size = int(epoch_size)
        self.quotas = {str(k): int(v) for k, v in quotas.items()}
        if set(self.quotas) != {"dynamic", "balanced", "static"} or sum(self.quotas.values()) != self.epoch_size:
            raise ValueError("dynamic/balanced/static quotas must sum to epoch_size")
        self.candidate_draws = int(candidate_draws)
        self.missing_history_quota = int(missing_history_quota)
        self.metadata_sha256 = _metadata_sha(metadata_path)
        rows = [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        scores = {row["file_name"]: float(row["temporal_novelty_score"]) for row in rows}
        availability = {row["file_name"]: bool(row.get("history_available", True)) for row in rows}
        missing = [record.file_name for record in data_source.records if record.file_name not in scores]
        if missing:
            raise ValueError(f"novelty metadata missing {len(missing)} training records")
        ranked = sorted((i for i, record in enumerate(data_source.records) if availability[record.file_name]),
                        key=lambda i: (scores[data_source.records[i].file_name], data_source.records[i].file_name))
        self.unavailable = [i for i, record in enumerate(data_source.records) if not availability[record.file_name]]
        n = len(ranked); static_end = round(.20 * n); dynamic_start = n - round(.50 * n)
        self.strata = {"static": ranked[:static_end], "balanced": ranked[static_end:dynamic_start], "dynamic": ranked[dynamic_start:]}
        self.history_available = torch.tensor(
            [availability[record.file_name] for record in data_source.records], dtype=torch.bool)
        if not 0 <= self.missing_history_quota <= self.quotas["static"]:
            raise ValueError("missing_history_quota must fit inside the static quota")
        if self.missing_history_quota > len(self.unavailable):
            raise ValueError("missing_history_quota exceeds unavailable-history pool")
        draw_quotas = dict(self.quotas); draw_quotas["static"] -= self.missing_history_quota
        for name, quota in draw_quotas.items():
            if quota > len(self.strata[name]): raise ValueError(f"quota {name} exceeds its stratum")
        labels = [list(record.action) + list(record.reason) for record in data_source.records]
        self.labels = torch.tensor(labels, dtype=torch.float64)
        frequency = self.labels.sum(0).clamp_min(1)
        inverse = (len(self.labels) / frequency).sqrt()
        positive_count = self.labels.sum(1).clamp_min(1)
        self.rarity = 1.0 + (self.labels * inverse).sum(1) / positive_count
        self.pool_prevalence = self.labels.mean(0)
        self.exposure = torch.zeros(len(data_source.records), dtype=torch.long)
        self.epoch = 0; self.consumed = 0
        self._permutation, self._stratum_by_index = self._make_permutation(0)

    def _draw(self, pool: list[int], quota: int, epoch: int, stratum_id: int) -> list[int]:
        if quota == 0:
            return []
        pool_indices = torch.tensor(pool, dtype=torch.long)
        # Exhaust the least-exposed tier before drawing avoidable repeats. This
        # preserves full-pool rotation while rarity only decides ties.
        mandatory_parts: list[torch.Tensor] = []
        boundary: torch.Tensor | None = None
        remaining = quota
        for level in sorted(self.exposure[pool_indices].unique().tolist()):
            tier = pool_indices[self.exposure[pool_indices] == level]
            if len(tier) <= remaining:
                mandatory_parts.append(tier)
                remaining -= len(tier)
            else:
                boundary = tier
                break
        mandatory = torch.cat(mandatory_parts) if mandatory_parts else torch.empty(0, dtype=torch.long)
        if remaining == 0:
            return mandatory.tolist()
        assert boundary is not None
        indices = boundary
        weights = self.rarity[indices] / (1.0 + self.exposure[indices].double())
        best: tuple[float, list[int]] | None = None
        for draw in range(self.candidate_draws):
            generator = torch.Generator().manual_seed(self.seed + epoch * 1009 + stratum_id * 101 + draw)
            local = torch.multinomial(weights, remaining, replacement=False, generator=generator)
            selected = torch.cat((mandatory, indices[local]))
            prevalence = self.labels[selected].mean(0)
            scale = self.pool_prevalence.clamp_min(.01).sqrt()
            distribution_error = float(((prevalence - self.pool_prevalence).abs() / scale).mean())
            exposure_error = float(self.exposure[selected].double().mean()) / max(1.0, float(self.exposure.max()))
            score = distribution_error + .05 * exposure_error
            candidate = selected.tolist()
            if best is None or score < best[0]: best = (score, candidate)
        assert best is not None
        return best[1]

    def _make_permutation(self, epoch: int) -> tuple[list[int], dict[int, str]]:
        selected: list[int] = []; owners: dict[int, str] = {}
        for stratum_id, name in enumerate(("dynamic", "balanced", "static")):
            quota = self.quotas[name] - (self.missing_history_quota if name == "static" else 0)
            chosen = self._draw(self.strata[name], quota, epoch, stratum_id)
            selected.extend(chosen); owners.update({index: name for index in chosen})
        missing = self._draw(self.unavailable, self.missing_history_quota, epoch, 3)
        selected.extend(missing); owners.update({index: "static" for index in missing})
        generator = torch.Generator().manual_seed(self.seed + epoch * 7919)
        order = torch.randperm(len(selected), generator=generator).tolist()
        return [selected[i] for i in order], owners

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for position in range(self.consumed, len(self._permutation)):
            index = self._permutation[position]
            yield index, self.seed + self.epoch * self.epoch_size + position

    def __len__(self) -> int: return len(self._permutation) - self.consumed
    @property
    def epoch_complete(self) -> bool: return self.consumed == self.epoch_size

    def mark_consumed(self, count: int) -> None:
        self.consumed += int(count)
        if self.consumed > self.epoch_size: raise RuntimeError("sampler consumed beyond epoch budget")

    def current_stats(self) -> dict[str, Any]:
        chosen = torch.tensor(self._permutation, dtype=torch.long)
        by_stratum = {}
        for name in self.quotas:
            owned = torch.tensor([i for i in self._permutation if self._stratum_by_index[i] == name], dtype=torch.long)
            by_stratum[name] = float(self.history_available[owned].float().mean())
        return {"epoch": self.epoch, "selected_count": len(chosen), "unique_count": len(set(self._permutation)),
                "stratum_counts": {name: sum(self._stratum_by_index[i] == name for i in self._permutation) for name in self.quotas},
                "positive_counts_action": self.labels[chosen, :4].sum(0).long().tolist(),
                "positive_counts_reason": self.labels[chosen, 4:].sum(0).long().tolist(),
                "mean_prior_exposure": float(self.exposure[chosen].float().mean()),
                "pool_covered_once_rate": float((self.exposure > 0).float().mean()),
                "history_available_count": int(self.history_available[chosen].sum()),
                "missing_history_count": int((~self.history_available[chosen]).sum()),
                "history_available_rate": float(self.history_available[chosen].float().mean()),
                "history_available_by_stratum": by_stratum,
                "metadata_sha256": self.metadata_sha256}

    def advance_epoch(self) -> None:
        if not self.epoch_complete: raise RuntimeError("cannot advance incomplete epoch")
        self.exposure[torch.tensor(self._permutation)] += 1
        self.epoch += 1; self.consumed = 0
        self._permutation, self._stratum_by_index = self._make_permutation(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "epoch": self.epoch, "consumed": self.consumed,
                "dataset_size": len(self.data_source), "epoch_size": self.epoch_size,
                "missing_history_quota": self.missing_history_quota,
                "metadata_sha256": self.metadata_sha256, "permutation": self._permutation,
                "exposure": self.exposure.tolist(), "stratum_by_index": self._stratum_by_index}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        identity = (int(state["seed"]), int(state["dataset_size"]), int(state["epoch_size"]),
                    int(state.get("missing_history_quota", 0)), state["metadata_sha256"])
        expected = (self.seed, len(self.data_source), self.epoch_size,
                    self.missing_history_quota, self.metadata_sha256)
        if identity != expected: raise RuntimeError("budget sampler identity mismatch")
        permutation = [int(x) for x in state["permutation"]]
        if len(permutation) != self.epoch_size or len(set(permutation)) != self.epoch_size:
            raise RuntimeError("invalid budget sampler permutation")
        self.epoch = int(state["epoch"]); self.consumed = int(state["consumed"])
        self._permutation = permutation; self.exposure = torch.tensor(state["exposure"], dtype=torch.long)
        self._stratum_by_index = {int(k): str(v) for k, v in state["stratum_by_index"].items()}
