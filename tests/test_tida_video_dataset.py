import json

import torch
from PIL import Image

from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset


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
    def decoder(_path, indices):
        return [Image.new("RGB", (32, 18), "white") for _ in indices], torch.ones(len(indices), dtype=torch.bool)
    sample = BDDOIAVideoDataset(manifest, "test", decoder=decoder)[0]
    assert sample["target_image"].shape == (3, 360, 640)
    assert sample["context_images"].shape == (14, 3, 192, 344)
    assert sample["timestamps"].shape == (15,) and sample["timestamps"][-1] == 0
    expected_actual = (sample["frame_indices"].float() - 150.0) / 30.0
    assert torch.equal(sample["timestamps"], expected_actual)
    assert not torch.equal(sample["timestamps"], sample["requested_timestamps"])
    assert sample["action"].shape == (4,) and sample["reason"].shape == (21,)
