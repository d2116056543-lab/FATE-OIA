from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from fate_oia.datasets.mosaic_icdor_split import make_icdor_train_splits
from fate_oia.engine import train_acpr_mosaic_trust_icdor as trainer


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


def test_factor_aware_audit_subset_balances_real_geometry_availability(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset(6)
    geometry = {
        "frame_0000.jpg": ("car", None),
        "frame_0001.jpg": ("traffic light", None),
        "frame_0002.jpg": (None, "drivable"),
        "frame_0003.jpg": ("car", "drivable"),
        "frame_0004.jpg": ("traffic light", None),
        "frame_0005.jpg": (None, None),
    }

    class _Index:
        def lookup(self, file_name: str) -> SimpleNamespace:
            label_json, drivable_map = geometry[file_name]
            return SimpleNamespace(label_json=label_json, drivable_map=drivable_map)

    monkeypatch.setattr(
        trainer,
        "load_bdd100k_objects",
        lambda label_json: [{"category": label_json}] if label_json else [],
    )

    selected = trainer._factor_aware_audit_subset(
        dataset, range(6), 3, grounding_index=_Index(), seed=20260713,
    )

    assert len(selected) == 3
    assert selected == trainer._factor_aware_audit_subset(
        dataset, range(6), 3, grounding_index=_Index(), seed=20260713,
    )
    selected_names = {dataset.samples[index].file_name for index in selected}
    assert any(geometry[name][0] == "car" for name in selected_names)
    assert any(geometry[name][0] == "traffic light" for name in selected_names)
    assert any(geometry[name][1] == "drivable" for name in selected_names)


def test_factor_aware_subset_is_not_filename_order_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset(8)
    geometry = {
        "frame_0000.jpg": (None, None),
        "frame_0001.jpg": (None, None),
        "frame_0002.jpg": (None, None),
        "frame_0003.jpg": (None, None),
        "frame_0004.jpg": ("car", None),
        "frame_0005.jpg": ("traffic light", None),
        "frame_0006.jpg": (None, "drivable"),
        "frame_0007.jpg": ("car", "drivable"),
    }

    class _Index:
        def lookup(self, file_name: str) -> SimpleNamespace:
            label_json, drivable_map = geometry[file_name]
            return SimpleNamespace(label_json=label_json, drivable_map=drivable_map)

    monkeypatch.setattr(
        trainer,
        "load_bdd100k_objects",
        lambda label_json: [{"category": label_json}] if label_json else [],
    )
    selected = trainer._factor_aware_audit_subset(
        dataset, range(8), 4, grounding_index=_Index(), seed=20260713,
    )
    assert set(selected) != set(range(4))
    selected_names = {dataset.samples[index].file_name for index in selected}
    assert any(geometry[name][0] == "car" for name in selected_names)
    assert any(geometry[name][0] == "traffic light" for name in selected_names)
    assert any(geometry[name][1] == "drivable" for name in selected_names)
