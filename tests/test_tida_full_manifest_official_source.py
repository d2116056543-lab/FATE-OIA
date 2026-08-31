import json
from dataclasses import dataclass

from PIL import Image

from fate_oia.engine.build_tida_full_manifest import discover_labeled_clips


@dataclass(frozen=True)
class OfficialSample:
    file_name: str
    image_path: str
    action: tuple[float, ...]
    reason: tuple[float, ...]


def _dataset_root(tmp_path, split="test"):
    root = tmp_path / "video"
    (root / "clips_prev5s" / split).mkdir(parents=True)
    (root / "labels" / split).mkdir(parents=True)
    (root / "last_frames" / split).mkdir(parents=True)
    return root


def test_official_sample_is_canonical_target_and_label_source(tmp_path):
    root = _dataset_root(tmp_path)
    clip = root / "clips_prev5s" / "test" / "case_1_prev5s.mp4"
    clip.write_bytes(b"video")
    generated = root / "last_frames" / "test" / "case_1_lastframe.jpg"
    Image.new("RGB", (8, 8), "red").save(generated)
    official = tmp_path / "official" / "case_1.jpg"
    official.parent.mkdir()
    Image.new("RGB", (8, 8), "green").save(official)
    wrong = {
        "image_name": "case_1.jpg",
        "last_frame_image_path": str(generated),
        "action_4": None,
        "reason_21": None,
    }
    (root / "labels" / "test" / "case_1_prev5s.json").write_text(
        json.dumps(wrong), encoding="utf-8"
    )
    sample = OfficialSample(
        "case_1.jpg", str(official), (1.0, 0.0, 1.0, 0.0), tuple([1.0] + [0.0] * 20)
    )

    rows, audit = discover_labeled_clips(
        root,
        official_samples_by_split={"train": {}, "test": {"case_1.jpg": sample}},
        probe=lambda _: (30.0, 151),
    )

    assert audit["accepted_by_split"]["test"] == 1
    assert rows[0].target_image_path == official
    assert rows[0].action == sample.action
    assert rows[0].reason == sample.reason
    assert rows[0].source_manifest_path == "official_bdd_oia_dataset"


def test_invalid_video_is_explicit_history_unavailable_not_dropped(tmp_path):
    root = _dataset_root(tmp_path)
    clip = root / "clips_prev5s" / "test" / "bad_prev5s.mp4"
    clip.write_bytes(b"bad")
    official = tmp_path / "bad.jpg"
    Image.new("RGB", (8, 8), "blue").save(official)
    sample = OfficialSample(
        "bad.jpg", str(official), (1.0, 0.0, 0.0, 0.0), tuple([0.0] * 21)
    )

    rows, audit = discover_labeled_clips(
        root,
        official_samples_by_split={"train": {}, "test": {"bad.jpg": sample}},
        probe=lambda _: (_ for _ in ()).throw(ValueError("broken")),
        allow_invalid_history=True,
    )

    assert len(rows) == 1
    assert rows[0].history_available is False
    assert rows[0].history_unavailable_reason == "invalid_video"
    assert audit["history_unavailable_by_split"]["test"] == 1
