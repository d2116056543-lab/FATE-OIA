from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fate_oia.datasets.mosaic_train_calib_split import (
    make_multilabel_train_calib_indices,
    verify_train_calib_split_artifacts,
)


@dataclass(frozen=True)
class _Sample:
    split: str
    file_name: str
    action: tuple[float, ...]
    reason: tuple[float, ...]


class _MetadataOnlyDataset:
    def __init__(self, samples: list[_Sample]) -> None:
        self.samples = samples
        self.getitem_calls = 0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        self.getitem_calls += 1
        raise AssertionError("split generation must not decode images")


class _WrongLengthDataset(_MetadataOnlyDataset):
    def __len__(self) -> int:
        return len(self.samples) + 1


def _dataset(count: int = 300, order: list[int] | None = None) -> _MetadataOnlyDataset:
    samples: list[_Sample] = []
    for index in range(count):
        action = [0.0] * 4
        reason = [0.0] * 21
        if index < 25:
            action = [1.0] * 4
            reason = [1.0] * 21
        else:
            action[index % 4] = 1.0
            reason[index % 21] = 1.0
            reason[(index * 7 + 3) % 21] = 1.0
        samples.append(_Sample("train", f"clip_{index:04d}.jpg", tuple(action), tuple(reason)))
    if order is not None:
        samples = [samples[index] for index in order]
    return _MetadataOnlyDataset(samples)


def test_multilabel_split_is_deterministic_stratified_and_metadata_only(tmp_path: Path) -> None:
    dataset = _dataset()
    main, calib = make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)

    assert len(calib) == 30
    assert len(main) == 270
    assert set(main).isdisjoint(calib)
    assert sorted(main + calib) == list(range(300))
    assert dataset.getitem_calls == 0

    for label_index in range(25):
        available = sum((sample.action + sample.reason)[label_index] for sample in dataset.samples)
        selected = sum((dataset.samples[index].action + dataset.samples[index].reason)[label_index] for index in calib)
        assert selected >= min(20, int(available))

    assert verify_train_calib_split_artifacts(tmp_path, dataset)


def test_multilabel_split_selected_file_names_do_not_depend_on_dataset_order() -> None:
    normal = _dataset()
    reverse_order = list(reversed(range(len(normal.samples))))
    reversed_dataset = _dataset(order=reverse_order)

    _, calib_a = make_multilabel_train_calib_indices(normal)
    _, calib_b = make_multilabel_train_calib_indices(reversed_dataset)
    names_a = {normal.samples[index].file_name for index in calib_a}
    names_b = {reversed_dataset.samples[index].file_name for index in calib_b}
    assert names_a == names_b


def test_multilabel_split_rejects_test_leakage_and_duplicate_ids() -> None:
    leaked = _dataset()
    leaked.samples[7] = _Sample("test", "test_clip.jpg", (1.0, 0.0, 0.0, 0.0), (0.0,) * 21)
    with pytest.raises(ValueError, match="train samples only"):
        make_multilabel_train_calib_indices(leaked)

    duplicated = _dataset()
    duplicated.samples[7] = _Sample(
        "train",
        duplicated.samples[6].file_name,
        duplicated.samples[7].action,
        duplicated.samples[7].reason,
    )
    with pytest.raises(ValueError, match="duplicate file_name"):
        make_multilabel_train_calib_indices(duplicated)


def test_multilabel_split_rejects_invalid_label_contract() -> None:
    malformed = _dataset()
    malformed.samples[0] = _Sample("train", "bad.jpg", (1.0, 0.0), (0.0,) * 21)
    with pytest.raises(ValueError, match="4 action and 21 reason"):
        make_multilabel_train_calib_indices(malformed)

    non_binary = _dataset()
    non_binary.samples[0] = _Sample("train", "bad_value.jpg", (0.5, 0.0, 0.0, 0.0), (0.0,) * 21)
    with pytest.raises(ValueError, match="binary"):
        make_multilabel_train_calib_indices(non_binary)


def test_split_artifact_hash_detects_tampering(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    expected_files = {
        "train_main_indices.json",
        "train_calib_indices.json",
        "train_calib_split_stats.json",
        "train_calib_split_hash.json",
    }
    assert expected_files <= {path.name for path in tmp_path.iterdir()}

    stats = json.loads((tmp_path / "train_calib_split_stats.json").read_text(encoding="utf-8"))
    assert stats["seed"] == 20260710
    assert stats["calib_fraction"] == 0.10
    assert stats["label_count"] == 25
    assert stats["all_targets_met"] is True

    calib_path = tmp_path / "train_calib_indices.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8"))
    calib[0], calib[1] = calib[1], calib[0]
    calib_path.write_text(json.dumps(calib), encoding="utf-8")
    with pytest.raises(ValueError, match="split artifact hash mismatch"):
        verify_train_calib_split_artifacts(tmp_path, dataset)


def test_split_hash_binds_label_metadata(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    original = dataset.samples[0]
    dataset.samples[0] = _Sample(
        original.split,
        original.file_name,
        (0.0, 0.0, 0.0, 0.0),
        original.reason,
    )
    with pytest.raises(ValueError, match="split artifact hash mismatch"):
        verify_train_calib_split_artifacts(tmp_path, dataset)


def test_split_rejects_dataset_length_metadata_mismatch() -> None:
    dataset = _WrongLengthDataset(_dataset().samples)
    with pytest.raises(ValueError, match="length does not match metadata"):
        make_multilabel_train_calib_indices(dataset)


def test_split_hash_binds_per_label_stats(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    stats_path = tmp_path / "train_calib_split_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["all_targets_met"] = False
    stats_path.write_text(json.dumps(stats), encoding="utf-8")
    with pytest.raises(ValueError, match="split stats hash mismatch"):
        verify_train_calib_split_artifacts(tmp_path, dataset)
