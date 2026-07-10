from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fate_oia.datasets.mosaic_train_calib_split import (
    _canonical_json_bytes,
    _repair_coverage_with_milp,
    _split_payload,
    _stable_rank,
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


class _SubsetLike:
    def __init__(self, dataset: _MetadataOnlyDataset, indices: list[object]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)


class _ShadowSamplesSubset(_SubsetLike):
    def __init__(self, dataset: _MetadataOnlyDataset, indices: list[object], samples: list[_Sample]) -> None:
        super().__init__(dataset, indices)
        self.samples = samples


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


def _coverage_trap_dataset() -> _MetadataOnlyDataset:
    samples: list[_Sample] = []
    patterns = [([0, 1, 2], 20), ([3, 4, 5], 20), ([0, 1, 3, 4], 20), ([], 340)]
    sample_id = 0
    for positive_labels, count in patterns:
        for _ in range(count):
            flat = [0.0] * 25
            for label in positive_labels:
                flat[label] = 1.0
            samples.append(
                _Sample("train", f"trap_{sample_id:04d}.jpg", tuple(flat[:4]), tuple(flat[4:]))
            )
            sample_id += 1
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


def test_split_uses_exact_feasibility_repair_when_greedy_misses_targets(tmp_path: Path) -> None:
    dataset = _coverage_trap_dataset()
    _, calib = make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    counts = [sum((dataset.samples[index].action + dataset.samples[index].reason)[label] for index in calib) for label in range(6)]
    assert counts == [20, 20, 20, 20, 20, 20]
    stats = json.loads((tmp_path / "train_calib_split_stats.json").read_text(encoding="utf-8"))
    assert stats["all_targets_met"] is True
    assert stats["coverage_solver"] == "scipy_milp_exact_repair"


def test_artifact_verifier_rejects_test_split_relabeling(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    original = dataset.samples[0]
    dataset.samples[0] = _Sample("test", original.file_name, original.action, original.reason)
    with pytest.raises(ValueError, match="train samples only"):
        verify_train_calib_split_artifacts(tmp_path, dataset)


def test_split_protocol_constants_are_not_user_overridable() -> None:
    dataset = _dataset()
    with pytest.raises(ValueError, match="protocol requires calib_fraction=0.10"):
        make_multilabel_train_calib_indices(dataset, calib_fraction=0.2)
    with pytest.raises(ValueError, match="protocol requires seed=20260710"):
        make_multilabel_train_calib_indices(dataset, seed=7)
    with pytest.raises(ValueError, match="protocol requires min_calib_positives=20"):
        make_multilabel_train_calib_indices(dataset, min_calib_positives=5)


def test_verifier_binds_protocol_fields_in_hash_record(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    record_path = tmp_path / "train_calib_split_hash.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["seed"] = 7
    record["calib_fraction"] = 0.2
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="split hash metadata mismatch"):
        verify_train_calib_split_artifacts(tmp_path, dataset)


def test_subset_indices_require_exact_non_negative_integers() -> None:
    base = _dataset()
    with pytest.raises(ValueError, match="subset indices must be exact non-negative integers"):
        make_multilabel_train_calib_indices(_SubsetLike(base, [0, 1.0, 2]))
    with pytest.raises(ValueError, match="subset indices must be exact non-negative integers"):
        make_multilabel_train_calib_indices(_SubsetLike(base, [0, -1, 2]))


def test_subset_metadata_cannot_be_shadowed_to_bypass_real_indices() -> None:
    base = _dataset()
    leaked = base.samples[0]
    base.samples[0] = _Sample("test", leaked.file_name, leaked.action, leaked.reason)
    shadow = _dataset().samples
    with pytest.raises(ValueError, match="train samples only"):
        make_multilabel_train_calib_indices(_ShadowSamplesSubset(base, list(range(len(base))), shadow))


def test_exact_repair_is_file_name_deterministic_across_input_orders() -> None:
    names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
    labels = [
        tuple([1, 0, 1, 0] + [0] * 21),
        tuple([1, 0, 0, 1] + [0] * 21),
        tuple([0, 1, 1, 0] + [0] * 21),
        tuple([0, 1, 0, 1] + [0] * 21),
    ]
    selected_name_sets: set[tuple[str, ...]] = set()
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
        ordered_names = [names[index] for index in order]
        ordered_labels = [labels[index] for index in order]
        selected, status = _repair_coverage_with_milp(
            labels=ordered_labels,
            targets=[1, 1, 1, 1] + [0] * 21,
            calib_count=2,
            greedy_selected=[ordered_names.index("a.jpg"), ordered_names.index("b.jpg")],
            ranks=[_stable_rank(20260710, name) for name in ordered_names],
            file_names=ordered_names,
        )
        assert selected is not None and status == "scipy_milp_exact_repair"
        selected_name_sets.add(tuple(sorted(ordered_names[index] for index in selected)))
    assert len(selected_name_sets) == 1


def test_verifier_recomputes_deterministic_partition_not_only_self_consistent_hashes(tmp_path: Path) -> None:
    dataset = _dataset()
    make_multilabel_train_calib_indices(dataset, output_dir=tmp_path)
    original_calib = json.loads((tmp_path / "train_calib_indices.json").read_text(encoding="utf-8"))
    replacement_calib = list(range(len(original_calib)))
    assert replacement_calib != original_calib
    replacement_main = [index for index in range(len(dataset)) if index not in set(replacement_calib)]
    (tmp_path / "train_main_indices.json").write_text(json.dumps(replacement_main), encoding="utf-8")
    (tmp_path / "train_calib_indices.json").write_text(json.dumps(replacement_calib), encoding="utf-8")

    stats = json.loads((tmp_path / "train_calib_split_stats.json").read_text(encoding="utf-8"))
    record_path = tmp_path / "train_calib_split_hash.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    file_names = [sample.file_name for sample in dataset.samples]
    labels = [tuple(int(value) for value in sample.action + sample.reason) for sample in dataset.samples]
    payload = _split_payload(
        seed=20260710,
        calib_fraction=0.10,
        file_names=file_names,
        label_sha256=record["label_sha256"],
        min_calib_positives=20,
        main_indices=replacement_main,
        calib_indices=replacement_calib,
    )
    import hashlib

    record["split_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest().upper()
    record["stats_sha256"] = hashlib.sha256(_canonical_json_bytes(stats)).hexdigest().upper()
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic protocol partition"):
        verify_train_calib_split_artifacts(tmp_path, dataset)
