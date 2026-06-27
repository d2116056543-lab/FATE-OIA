from __future__ import annotations

import pickle
from pathlib import Path

from PIL import Image

from fate_oia.acpr_interactflow.psi_damo_dataset import PSIDAMO11902Dataset


def _make_package(tmp_path: Path) -> tuple[Path, Path]:
    pkg = tmp_path / "pkg"
    frames = tmp_path / "frames"
    (pkg / "samples").mkdir(parents=True)
    (pkg / "reason_exp29").mkdir(parents=True)
    frames.mkdir()
    frame_names = []
    for i in range(16):
        name = f"video/frame_{i:03d}.jpg"
        path = frames / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (i, i, i)).save(path)
        if i < 15:
            frame_names.append(name)
    sample = {
        "input_frames": frame_names,
        "target_frame": "video/frame_015.jpg",
        "action_soft": [0.1, 0.2, 0.7],
        "action_majority": 2,
        "exp29": [0] * 29,
        "sample_id": "s0",
    }
    with (pkg / "samples" / "train.pkl").open("wb") as f:
        pickle.dump([sample], f)
    return pkg, frames


def test_dataset_uses_15_observed_frames_and_masks_all_zero_exp29(tmp_path: Path):
    pkg, frames = _make_package(tmp_path)
    ds = PSIDAMO11902Dataset(pkg, "train", frames_root=frames, image_size=(32, 64))
    item = ds[0]
    assert item["input_frames"].shape == (15, 3, 32, 64)
    assert "frame_015" not in " ".join(item["frame_paths"])
    assert item["target_frame_path"].endswith("frame_015.jpg")
    assert item["exp29_mask"].sum().item() == 0

