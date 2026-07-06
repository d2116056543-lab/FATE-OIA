from __future__ import annotations

import pickle
import json
from pathlib import Path

import pytest
from PIL import Image


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color).save(path)


def _write_package(root: Path, frames_root: Path) -> None:
    (root / "samples").mkdir(parents=True)
    (root / "reason_exp29").mkdir(parents=True)
    for frame_id in range(1, 17):
        _make_image(frames_root / "frames" / "video_0001" / f"{frame_id:03d}.jpg", (frame_id, 2 * frame_id, 3 * frame_id))
    sample = {
        "video_id": "video_0001",
        "input_frames": list(range(1, 16)),
        "target_frame": 16,
        "action_name": "stop_car",
        "action_soft_target": [0.0, 0.0, 1.0],
        "paper_effective_weight": 1.0,
    }
    for split in ("train", "test"):
        with (root / "samples" / f"{split}.pkl").open("wb") as f:
            pickle.dump([sample], f)
        with (root / "reason_exp29" / f"{split}.pkl").open("wb") as f:
            pickle.dump({"labels": [[0.0] * 28 + [1.0]], "masks": [1.0]}, f)


def test_sanity_dataset_exposes_target_last_and_clip_modes(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset

    package_root = tmp_path / "pkg"
    frames_root = tmp_path / "PSI_data"
    _write_package(package_root, frames_root)

    target = PSISanityDataset(package_root, "test", input_mode="target_frame", frames_root=frames_root, image_size=(16, 24))[0]
    last = PSISanityDataset(package_root, "test", input_mode="last_observed", frames_root=frames_root, image_size=(16, 24))[0]
    clip = PSISanityDataset(package_root, "test", input_mode="clip15", frames_root=frames_root, image_size=(16, 24))[0]
    k_current = PSISanityDataset(package_root, "test", input_mode="k_current_15", frames_root=frames_root, image_size=(16, 24))[0]

    assert target["images"].shape == (3, 16, 24)
    assert last["images"].shape == (3, 16, 24)
    assert clip["images"].shape == (15, 3, 16, 24)
    assert k_current["images"].shape == (15, 3, 16, 24)
    assert target["input_mode"] == "target_frame"
    assert last["selected_frame_id"].item() == 15
    assert k_current["selected_frame_id"].item() == 16
    assert target["selected_frame_id"].item() == 16
    assert target["target_frame_path"] not in target["frame_paths"]
    assert k_current["target_frame_path"] in k_current["frame_paths"]
    assert clip["action_soft"].tolist() == [0.0, 0.0, 1.0]


def test_sanity_dataset_consumes_protocol_index_and_exp_policy(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset

    package_root = tmp_path / "pkg"
    frames_root = tmp_path / "PSI_data"
    _write_package(package_root, frames_root)

    records_path = package_root / "samples" / "train.pkl"
    with records_path.open("rb") as f:
        sample = pickle.load(f)[0]
    samples = []
    for offset in (0, 100):
        row = dict(sample)
        row["sample_id"] = f"row_{offset}"
        row["target_frame"] = 16
        row["decision_keyframe"] = 10 + offset
        row["explanation_keyframe"] = 20 + offset
        row["action_name"] = "reduce_speed"
        row["reasoning_text"] = "A pedestrian is walking into the lane near the ego vehicle."
        samples.append(row)
    with records_path.open("wb") as f:
        pickle.dump(samples, f)
    with (package_root / "reason_exp29" / "train.pkl").open("wb") as f:
        pickle.dump(
            {
                "labels": [[0.0] * 29, [1.0] + [0.0] * 28],
                "masks": [[1.0] * 29, [1.0] * 29],
            },
            f,
        )

    protocol_dir = tmp_path / "protocols" / "gap_decay_180"
    protocol_dir.mkdir(parents=True)
    with (protocol_dir / "train_indices.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source_index": 1, "action_weight": 0.5, "exp29_supervised": True}) + "\n")

    ds = PSISanityDataset(
        package_root,
        "train",
        input_mode="target_frame",
        frames_root=frames_root,
        image_size=(16, 24),
        protocol_index_dir=tmp_path / "protocols",
        protocol_name="gap_decay_180",
        exp_supervision_policy="near_keyframe_raw_mask",
        exp_near_keyframe_max_gap=60,
    )

    item = ds[0]
    assert len(ds) == 1
    assert item["sample_id"] == "row_100"
    assert item["paper_effective_weight"].item() == 0.5
    assert item["exp29"][0].item() == 1.0
    assert item["exp29_mask"].sum().item() == 29.0


def test_sanity_dataset_balanced_action_sampling(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset

    package_root = tmp_path / "pkg"
    frames_root = tmp_path / "PSI_data"
    _write_package(package_root, frames_root)
    records = []
    for idx, action in enumerate(
        [
            ("maintain_speed", [1.0, 0.0, 0.0]),
            ("maintain_speed", [1.0, 0.0, 0.0]),
            ("reduce_speed", [0.0, 1.0, 0.0]),
            ("stop_car", [0.0, 0.0, 1.0]),
        ]
    ):
        action_name, soft = action
        row = {
            "video_id": "video_0001",
            "input_frames": list(range(1, 16)),
            "target_frame": 16,
            "action_name": action_name,
            "action_soft_target": soft,
            "sample_id": f"row_{idx}",
        }
        records.append(row)
    for split in ("train", "test"):
        with (package_root / "samples" / f"{split}.pkl").open("wb") as f:
            pickle.dump(records, f)
        with (package_root / "reason_exp29" / f"{split}.pkl").open("wb") as f:
            pickle.dump({"labels": [[0.0] * 29 for _ in records], "masks": [0.0] * len(records)}, f)

    ds = PSISanityDataset(
        package_root,
        "train",
        input_mode="target_frame",
        frames_root=frames_root,
        image_size=(16, 24),
        max_samples=3,
        max_sample_strategy="balanced_action",
        max_sample_seed=0,
    )

    sampled = {ds[idx]["action_majority"].item() for idx in range(len(ds))}
    assert sampled == {0, 1, 2}


def test_sanity_dataset_uses_protocol_action_hard_for_metrics(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset

    package_root = tmp_path / "pkg"
    frames_root = tmp_path / "PSI_data"
    _write_package(package_root, frames_root)

    records_path = package_root / "samples" / "train.pkl"
    with records_path.open("rb") as f:
        sample = pickle.load(f)[0]
    sample = dict(sample)
    sample["sample_id"] = "tie_row"
    sample["action_name"] = "reduce_speed"
    sample["action_soft_target"] = [0.5, 0.5, 0.0]
    sample.pop("action_majority", None)
    sample.pop("action_label", None)
    with records_path.open("wb") as f:
        pickle.dump([sample], f)

    protocol_dir = tmp_path / "protocols" / "gap_decay_180"
    protocol_dir.mkdir(parents=True)
    with (protocol_dir / "train_indices.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"source_index": 0, "action_hard": 0, "action_name": "maintain_speed", "action_weight": 0.7}) + "\n")

    ds = PSISanityDataset(
        package_root,
        "train",
        input_mode="target_frame",
        frames_root=frames_root,
        image_size=(16, 24),
        protocol_index_dir=tmp_path / "protocols",
        protocol_name="gap_decay_180",
    )

    item = ds[0]
    assert item["action_majority"].item() == 0
    assert abs(item["paper_effective_weight"].item() - 0.7) < 1e-6


def test_sanity_dataset_can_weight_rows_by_decision_group(tmp_path: Path) -> None:
    from fate_oia.acpr_interactflow.psi_sanity_dataset import PSISanityDataset

    package_root = tmp_path / "pkg"
    frames_root = tmp_path / "PSI_data"
    _write_package(package_root, frames_root)

    records = []
    for idx, (decision_keyframe, action_name, soft) in enumerate(
        [
            (10, "maintain_speed", [1.0, 0.0, 0.0]),
            (10, "maintain_speed", [1.0, 0.0, 0.0]),
            (10, "maintain_speed", [1.0, 0.0, 0.0]),
            (30, "reduce_speed", [0.0, 1.0, 0.0]),
        ]
    ):
        records.append(
            {
                "video_id": "video_0001",
                "input_frames": list(range(1, 16)),
                "target_frame": 16,
                "decision_keyframe": decision_keyframe,
                "action_name": action_name,
                "action_soft_target": soft,
                "sample_id": f"row_{idx}",
                "paper_effective_weight": 1.0,
            }
        )
    with (package_root / "samples" / "train.pkl").open("wb") as f:
        pickle.dump(records, f)
    with (package_root / "reason_exp29" / "train.pkl").open("wb") as f:
        pickle.dump({"labels": [[0.0] * 29 for _ in records], "masks": [0.0] * len(records)}, f)

    ds = PSISanityDataset(
        package_root,
        "train",
        input_mode="target_frame",
        frames_root=frames_root,
        image_size=(16, 24),
        use_decision_group_weight=True,
    )

    weights = [ds[idx]["paper_effective_weight"].item() for idx in range(len(ds))]
    assert weights[:3] == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    assert weights[3] == pytest.approx(1.0)


