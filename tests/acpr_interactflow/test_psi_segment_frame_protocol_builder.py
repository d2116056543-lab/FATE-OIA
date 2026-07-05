from __future__ import annotations

import pickle
from pathlib import Path


def _write_source_package(root: Path) -> None:
    (root / "samples").mkdir(parents=True)
    (root / "reason_exp29").mkdir(parents=True)
    train_rows = []
    sample_id = 0
    for video_id, decision_keyframe, action_name, soft, offsets in [
        ("video_a", 100, "maintain_speed", [1.0, 0.0, 0.0], [0, 1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 35]),
        ("video_a", 200, "reduce_speed", [0.0, 1.0, 0.0], [0, 1, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),
        ("video_b", 50, "stop_car", [0.0, 0.0, 1.0], [0, 2, 4, 8, 12, 16, 20, 24, 28, 30]),
    ]:
        for offset in offsets:
            target = decision_keyframe + offset
            train_rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": video_id,
                    "input_frames": list(range(target - 15, target)),
                    "target_frame": target,
                    "decision_keyframe": decision_keyframe,
                    "explanation_keyframe": target if offset in (0, 3) else decision_keyframe + 100,
                    "reasoning_text": "aligned reason text" if offset in (0, 3) else "future filled reason text",
                    "explanation_text": "aligned scene text" if offset in (0, 3) else "future filled scene text",
                    "action_name": action_name,
                    "action_soft_target": soft,
                    "paper_effective_weight": 1.0,
                }
            )
            sample_id += 1

    # Duplicate visual target should not be allowed to leak across train/test.
    duplicate = dict(train_rows[1])
    duplicate["sample_id"] = "duplicate_target"
    duplicate["action_name"] = "maintain_speed"
    train_rows.append(duplicate)

    with (root / "samples" / "train.pkl").open("wb") as handle:
        pickle.dump(train_rows, handle)
    with (root / "samples" / "val.pkl").open("wb") as handle:
        pickle.dump([], handle)
    with (root / "samples" / "test.pkl").open("wb") as handle:
        pickle.dump([], handle)

    for split, count in [("train", len(train_rows)), ("val", 0), ("test", 0)]:
        with (root / "reason_exp29" / f"{split}.pkl").open("wb") as handle:
            pickle.dump({"labels": [[1.0] + [0.0] * 28 for _ in range(count)], "masks": [1.0] * count}, handle)


def _load_rows(root: Path, split: str) -> list[dict]:
    with (root / "samples" / f"{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def _load_exp(root: Path, split: str) -> list[dict]:
    with (root / "reason_exp29" / f"{split}.pkl").open("rb") as handle:
        return pickle.load(handle)


def _target_key(row: dict) -> tuple[str, int]:
    return str(row["video_id"]), int(row["target_frame"])


def _decision_key(row: dict) -> tuple[str, int]:
    return str(row["video_id"]), int(row["decision_keyframe"])


def test_segment_frame_protocol_builder_creates_leak_free_gap30_action_package(tmp_path: Path) -> None:
    from fate_oia.engine.build_psi_segment_frame_protocol import build_segment_frame_protocol

    source = tmp_path / "source"
    out_root = tmp_path / "out"
    _write_source_package(source)

    summary = build_segment_frame_protocol(
        source_package_root=source,
        output_root=out_root,
        protocol_name="segment_frame_split_gap30_stop20",
        max_gap=30,
        test_fraction=0.25,
        min_test_gap=3,
        target_stop_train_rate=0.20,
        seed=7,
    )

    package = out_root / "segment_frame_split_gap30_stop20"
    train = _load_rows(package, "train")
    test = _load_rows(package, "test")
    val = _load_rows(package, "val")
    train_exp = _load_exp(package, "train")
    test_exp = _load_exp(package, "test")

    assert val == []
    assert train
    assert test
    assert summary["leakage"]["target_frame_overlap_train_test"] == 0
    assert {_target_key(row) for row in train}.isdisjoint({_target_key(row) for row in test})
    assert {_decision_key(row) for row in train} & {_decision_key(row) for row in test}
    assert all(0 <= int(row["target_frame"]) - int(row["decision_keyframe"]) <= 30 for row in train + test)
    assert all(int(row["target_frame"]) - int(row["decision_keyframe"]) >= 3 for row in test)
    assert all(row["protocol_name"] == "segment_frame_split_gap30_stop20" for row in train + test)
    assert all(row["exp29_supervised"] is False for row in train + test)
    assert all(sum(item["exp29_mask"]) == 0.0 for item in train_exp + test_exp)

    stop_rate = sum(row["action_name"] == "stop_car" for row in train) / len(train)
    assert stop_rate >= 0.20

    summary_path = package / "summary.json"
    assert summary_path.exists()
    assert summary["splits"]["train"]["rows"] == len(train)
    assert summary["splits"]["test"]["rows"] == len(test)
    assert summary["exp29_policy"] == "all_unknown"


def test_segment_frame_protocol_builder_marks_only_temporally_aligned_exp_rows(tmp_path: Path) -> None:
    from fate_oia.engine.build_psi_segment_frame_protocol import build_segment_frame_protocol

    source = tmp_path / "source"
    out_root = tmp_path / "out"
    _write_source_package(source)

    summary = build_segment_frame_protocol(
        source_package_root=source,
        output_root=out_root,
        protocol_name="segment_frame_split_gap30_stop20_exp_aligned",
        max_gap=30,
        test_fraction=0.25,
        min_test_gap=3,
        target_stop_train_rate=0.20,
        seed=7,
        exp29_policy="aligned_only",
        exp29_alignment_window=3,
    )

    package = out_root / "segment_frame_split_gap30_stop20_exp_aligned"
    train = _load_rows(package, "train")
    test = _load_rows(package, "test")
    train_exp = _load_exp(package, "train")
    test_exp = _load_exp(package, "test")

    assert summary["exp29_policy"] == "aligned_only"
    assert summary["exp29_alignment_window"] == 3
    total_aligned = summary["splits"]["train"]["exp29_aligned_rows"] + summary["splits"]["test"]["exp29_aligned_rows"]
    assert total_aligned > 0
    assert summary["splits"]["train"]["exp29_supervised_rows"] == summary["splits"]["train"]["exp29_aligned_rows"]
    assert summary["splits"]["test"]["exp29_supervised_rows"] == summary["splits"]["test"]["exp29_aligned_rows"]

    saw_aligned = False
    saw_unknown = False
    for row, exp in zip(train + test, train_exp + test_exp):
        expected_aligned = abs(int(row["explanation_keyframe"]) - int(row["target_frame"])) <= 3
        assert row["exp29_temporally_aligned"] is expected_aligned
        assert row["exp29_supervised"] is expected_aligned
        assert bool(exp.get("exp29_aligned_text", False)) is expected_aligned
        if expected_aligned:
            saw_aligned = True
            assert sum(exp["exp29_mask"]) == 29.0
            assert exp["exp29"][0] == 1.0
        else:
            saw_unknown = True
            assert sum(exp["exp29_mask"]) == 0.0
    assert saw_aligned
    assert saw_unknown
