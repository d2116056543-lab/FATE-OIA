import json

import torch
from PIL import Image

from fate_oia.datasets.bdd_oia_video import (
    BDDOIAVideoDataset,
    _decode_selected_frames_from_capture,
    quadratic_multirate_timestamps,
)
from fate_oia.transforms_video import SynchronizedVideoTransform


def test_video_dataset_returns_formal_tensor_contract(tmp_path):
    target = tmp_path / "target.jpg"; Image.new("RGB", (32, 18), "white").save(target)
    clip = tmp_path / "clip.mp4"; clip.write_bytes(b"stub")
    row = {
        "official_split": "test", "partition": "test", "file_name": "x.jpg",
        "target_image_path": str(target), "clip_path": str(clip), "source_video_id": "x",
        "duration_seconds": 5.0, "fps": 30.0, "num_frames": 151,
        "target_timestamp_seconds": 5.0, "target_frame_index": 150,
        "action": [1, 0, 0, 0], "reason": [0] * 21,
    }
    manifest = tmp_path / "manifest.jsonl"; manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    decoded_indices = []
    def decoder(_path, indices):
        decoded_indices.extend(indices.tolist())
        return [Image.new("RGB", (32, 18), "white") for _ in indices], torch.ones(len(indices), dtype=torch.bool)
    sample = BDDOIAVideoDataset(manifest, "test", decoder=decoder)[0]
    assert sample["target_image"].shape == (3, 360, 640)
    assert sample["context_images"].shape == (14, 3, 192, 344)
    assert sample["timestamps"].shape == (15,) and sample["timestamps"][-1] == 0
    expected_actual = (sample["frame_indices"].float() - 150.0) / 30.0
    assert torch.equal(sample["timestamps"], expected_actual)
    assert not torch.equal(sample["timestamps"], sample["requested_timestamps"])
    assert sample["action"].shape == (4,) and sample["reason"].shape == (21,)
    assert len(decoded_indices) == 14
    assert decoded_indices == sample["frame_indices"][:-1].tolist()
    assert 150 not in decoded_indices
    assert sample["frame_valid_mask"].tolist()[-1] is True


def test_sparse_capture_seeks_large_gaps_and_retrieves_only_requested_frames():
    class FakeCapture:
        def __init__(self):
            self.position = -1
            self.grab_count = 0
            self.retrieve_positions = []
            self.seek_positions = []

        def isOpened(self):
            return self.position < 10

        def grab(self):
            self.position += 1
            self.grab_count += 1
            return self.position <= 10

        def retrieve(self):
            import numpy as np

            self.retrieve_positions.append(self.position)
            frame = np.full((2, 3, 3), self.position, dtype=np.uint8)
            return True, frame

        def seek(self, position):
            self.position = position - 1
            self.seek_positions.append(position)
            return True

        def release(self):
            return None

    capture = FakeCapture()
    frames, valid = _decode_selected_frames_from_capture(
        capture,
        torch.tensor([1, 4, 9]),
        bgr_to_rgb=lambda frame: frame,
        seek_frame=lambda stream, position: stream.seek(position),
        sequential_gap=3,
    )
    assert capture.grab_count == 5
    assert capture.seek_positions == [1, 9]
    assert capture.retrieve_positions == [1, 4, 9]
    assert len(frames) == 3
    assert valid.tolist() == [True, True, True]


def test_precomputed_tracks_disable_timestamp_jitter_and_follow_horizontal_flip(tmp_path):
    target = tmp_path / "target.jpg"; Image.new("RGB", (32, 18), "white").save(target)
    clip = tmp_path / "clip.mp4"; clip.write_bytes(b"stub")
    row = {
        "official_split": "train", "partition": "train_core", "file_name": "x.jpg",
        "target_image_path": str(target), "clip_path": str(clip), "source_video_id": "x",
        "duration_seconds": 5.0, "fps": 30.0, "num_frames": 151,
        "target_timestamp_seconds": 5.0, "target_frame_index": 150,
        "action": [1, 0, 0, 0], "reason": [0] * 21,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    xy = torch.zeros(1, 15, 2, 2); xy[..., 0] = 0.25
    store = tmp_path / "tracks.pt"
    torch.save({"file_names": ["x.jpg"], "tracks_xy": xy,
                "visibility": torch.ones(1, 15, 2, dtype=torch.bool)}, store)
    decoder = lambda _path, indices: (
        [Image.new("RGB", (32, 18), "white") for _ in indices],
        torch.ones(len(indices), dtype=torch.bool),
    )
    dataset = BDDOIAVideoDataset(
        manifest, "train_core", training=True, decoder=decoder,
        transform=SynchronizedVideoTransform(flip_probability=1.0),
        object_track_store_path=store,
    )

    sample = dataset[0]

    assert dataset.object_tracks is None
    assert dataset.object_tracks_xy is not None
    assert dataset.object_tracks_visibility is not None
    assert dataset.object_tracks_xy.is_contiguous()
    assert dataset.object_tracks_visibility.is_contiguous()
    assert dataset.object_tracks_xy.is_shared()
    assert dataset.object_tracks_visibility.is_shared()
    assert torch.equal(sample["requested_timestamps"], quadratic_multirate_timestamps())
    assert torch.all(sample["object_tracks_xy"][..., 0] == -0.25)
    assert sample["object_tracks_visibility"].all()


def test_raw_frame_store_bypasses_native_video_decoder(tmp_path):
    target = tmp_path / "target.jpg"; Image.new("RGB", (32, 18), "white").save(target)
    clip = tmp_path / "clip.mp4"; clip.write_bytes(b"stub")
    row = {
        "official_split": "test", "partition": "test", "file_name": "x.jpg",
        "target_image_path": str(target), "clip_path": str(clip), "source_video_id": "x",
        "duration_seconds": 5.0, "fps": 30.0, "num_frames": 151,
        "target_timestamp_seconds": 5.0, "target_frame_index": 150,
        "action": [1, 0, 0, 0], "reason": [0] * 21,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    case_dir = tmp_path / "frames" / "test" / "x"; case_dir.mkdir(parents=True)
    for position in range(14):
        Image.new("RGB", (32, 18), (position, 0, 0)).save(case_dir / f"{position:02d}.jpg")

    def forbidden_decoder(_path, _indices):
        raise AssertionError("native video decoder must not be called")

    sample = BDDOIAVideoDataset(
        manifest, "test", decoder=forbidden_decoder, frame_store_root=tmp_path / "frames"
    )[0]

    assert sample["context_images"].shape == (14, 3, 192, 344)
    assert sample["frame_valid_mask"].all()


def test_raw_frame_store_uses_semicolon_separated_fallback_roots(tmp_path):
    target = tmp_path / "target.jpg"; Image.new("RGB", (32, 18), "white").save(target)
    clip = tmp_path / "clip.mp4"; clip.write_bytes(b"stub")
    row = {
        "official_split": "train", "partition": "train_core", "file_name": "fallback.jpg",
        "target_image_path": str(target), "clip_path": str(clip), "source_video_id": "fallback",
        "duration_seconds": 5.0, "fps": 30.0, "num_frames": 151,
        "target_timestamp_seconds": 5.0, "target_frame_index": 150,
        "action": [1, 0, 0, 0], "reason": [0] * 21,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    first = tmp_path / "frames-a"; first.mkdir()
    second = tmp_path / "frames-b"
    case_dir = second / "train_audit" / "fallback"; case_dir.mkdir(parents=True)
    for position in range(14):
        Image.new("RGB", (32, 18), (position, 0, 0)).save(case_dir / f"{position:02d}.jpg")

    def forbidden_decoder(_path, _indices):
        raise AssertionError("fallback raw-frame root must bypass video decoding")

    dataset = BDDOIAVideoDataset(
        manifest, "train_core", decoder=forbidden_decoder,
        frame_store_root=f"{first};{second}",
    )
    sample = dataset[0]

    assert len(dataset.frame_store_roots) == 2
    assert sample["context_images"].shape == (14, 3, 192, 344)
