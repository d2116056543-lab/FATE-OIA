from __future__ import annotations

import pickle
from pathlib import Path


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _sample(video: str, decision: int, target: int, action: str, soft: list[float]) -> dict:
    return {
        "video_id": video,
        "decision_keyframe": decision,
        "target_frame": target,
        "input_frames": list(range(max(0, target - 15), target)),
        "action_name": action,
        "action_soft_target": soft,
        "reasoning_text": f"{action} because of event {decision}",
    }


def _event_keys(rows: list[dict]) -> set[str]:
    return {f"{row['video_id']}::{row['decision_keyframe']}" for row in rows}


def test_event_disjoint_builder_preserves_expanded_rows_and_blocks_event_leakage(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_event_protocol import build_event_disjoint_package

    source = tmp_path / "source"
    rows = [
        _sample("video_a", 10, 11, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_a", 10, 12, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_a", 30, 31, "reduce_speed", [0.0, 1.0, 0.0]),
        _sample("video_a", 30, 32, "reduce_speed", [0.0, 1.0, 0.0]),
        _sample("video_b", 50, 51, "stop_car", [0.0, 0.0, 1.0]),
        _sample("video_b", 50, 52, "stop_car", [0.0, 0.0, 1.0]),
        _sample("video_c", 70, 71, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_c", 90, 91, "reduce_speed", [0.0, 1.0, 0.0]),
    ]
    labels = [[1.0] + [0.0] * 28 for _ in rows]
    masks = [1.0 for _ in rows]
    _dump(source / "samples" / "train.pkl", rows[:4])
    _dump(source / "samples" / "val.pkl", rows[4:6])
    _dump(source / "samples" / "test.pkl", rows[6:])
    _dump(source / "reason_exp29" / "train.pkl", {"labels": labels[:4], "masks": masks[:4]})
    _dump(source / "reason_exp29" / "val.pkl", {"labels": labels[4:6], "masks": masks[4:6]})
    _dump(source / "reason_exp29" / "test.pkl", {"labels": labels[6:], "masks": masks[6:]})

    out = tmp_path / "event_pkg"
    summary = build_event_disjoint_package(
        source,
        out,
        fold=0,
        num_folds=2,
        dev_fraction=0.25,
        seed=13,
    )

    split_rows = {}
    for split in ("train", "val", "test"):
        with (out / "samples" / f"{split}.pkl").open("rb") as handle:
            split_rows[split] = pickle.load(handle)

    train_events = _event_keys(split_rows["train"])
    dev_events = _event_keys(split_rows["val"])
    test_events = _event_keys(split_rows["test"])
    assert train_events.isdisjoint(dev_events)
    assert train_events.isdisjoint(test_events)
    assert dev_events.isdisjoint(test_events)

    # Expanded rows stay together: every event appears in exactly one split with all its rows.
    total_output_rows = sum(len(value) for value in split_rows.values())
    assert total_output_rows == len(rows)
    assert summary["shared_events"] == 0
    assert summary["total_events"] == 5
    assert summary["splits"]["test"]["events"] > 0
    assert summary["splits"]["val"]["events"] > 0


def test_event_disjoint_builder_filters_large_target_decision_gap_without_event_leakage(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_event_protocol import build_event_disjoint_package

    source = tmp_path / "source_gap"
    rows = [
        _sample("video_a", 10, 11, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_a", 10, 130, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_b", 30, 31, "reduce_speed", [0.0, 1.0, 0.0]),
        _sample("video_b", 30, 160, "reduce_speed", [0.0, 1.0, 0.0]),
        _sample("video_c", 50, 51, "stop_car", [0.0, 0.0, 1.0]),
        _sample("video_c", 50, 180, "stop_car", [0.0, 0.0, 1.0]),
        _sample("video_d", 70, 71, "maintain_speed", [1.0, 0.0, 0.0]),
        _sample("video_e", 90, 91, "reduce_speed", [0.0, 1.0, 0.0]),
    ]
    labels = [[1.0] + [0.0] * 28 for _ in rows]
    masks = [1.0 for _ in rows]
    _dump(source / "samples" / "train.pkl", rows[:3])
    _dump(source / "samples" / "val.pkl", rows[3:5])
    _dump(source / "samples" / "test.pkl", rows[5:])
    _dump(source / "reason_exp29" / "train.pkl", {"labels": labels[:3], "masks": masks[:3]})
    _dump(source / "reason_exp29" / "val.pkl", {"labels": labels[3:5], "masks": masks[3:5]})
    _dump(source / "reason_exp29" / "test.pkl", {"labels": labels[5:], "masks": masks[5:]})

    out = tmp_path / "event_pkg_gap30"
    summary = build_event_disjoint_package(
        source,
        out,
        fold=0,
        num_folds=2,
        dev_fraction=0.25,
        seed=19,
        max_target_decision_gap=30,
    )

    split_rows = {}
    for split in ("train", "val", "test"):
        with (out / "samples" / f"{split}.pkl").open("rb") as handle:
            split_rows[split] = pickle.load(handle)

    all_rows = [row for rows_for_split in split_rows.values() for row in rows_for_split]
    assert len(all_rows) == 5
    assert all(0 <= row["target_frame"] - row["decision_keyframe"] <= 30 for row in all_rows)
    assert summary["source_rows"] == 8
    assert summary["filtered_rows"] == 5
    assert summary["max_target_decision_gap"] == 30

    split_events = {split: _event_keys(rows_for_split) for split, rows_for_split in split_rows.items()}
    assert split_events["train"].isdisjoint(split_events["val"])
    assert split_events["train"].isdisjoint(split_events["test"])
    assert split_events["val"].isdisjoint(split_events["test"])
