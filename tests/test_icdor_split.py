from __future__ import annotations

from dataclasses import dataclass

import pytest

from fate_oia.datasets.mosaic_icdor_split import make_icdor_train_splits


@dataclass(frozen=True)
class _Sample:
    split: str
    file_name: str
    action: tuple[float, ...]
    reason: tuple[float, ...]


class _Dataset:
    def __init__(self, samples: list[_Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> object:
        raise AssertionError("IC-DOR split construction must use metadata only")


def _dataset(count: int = 200) -> _Dataset:
    samples: list[_Sample] = []
    for index in range(count):
        action = [0.0] * 4
        reason = [0.0] * 21
        action[index % 4] = 1.0
        reason[index % 21] = 1.0
        reason[(index * 5 + 3) % 21] = 1.0
        samples.append(_Sample("train", f"frame_{index:04d}.jpg", tuple(action), tuple(reason)))
    return _Dataset(samples)


def test_icdor_split_is_disjoint_deterministic_and_complete() -> None:
    dataset = _dataset()
    first = make_icdor_train_splits(dataset, seed=20260713)
    second = make_icdor_train_splits(dataset, seed=20260713)

    assert first == second
    core = set(first.train_core_indices)
    audit = set(first.train_audit_indices)
    calib = set(first.train_calib_indices)
    assert not core & audit
    assert not core & calib
    assert not audit & calib
    assert core | audit | calib == set(range(len(dataset)))
    assert len(core) == 160
    assert len(audit) == 20
    assert len(first.audit_visual_indices) == 10
    assert len(first.audit_target_indices) == 10
    assert set(first.audit_visual_indices).isdisjoint(first.audit_target_indices)
    assert set(first.audit_visual_indices) | set(first.audit_target_indices) == audit
    assert len(calib) == 20
    assert first.seed == 20260713
    assert len(first.split_sha256) == 64
    assert len(first.label_positive_counts) == 25


def test_icdor_split_rejects_non_train_metadata() -> None:
    dataset = _dataset()
    sample = dataset.samples[0]
    dataset.samples[0] = _Sample("test", sample.file_name, sample.action, sample.reason)
    with pytest.raises(ValueError, match="train samples only"):
        make_icdor_train_splits(dataset, seed=20260713)
