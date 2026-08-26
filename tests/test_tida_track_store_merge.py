import json

import torch

from fate_oia.engine.merge_tida_object_track_stores import merge_track_stores


def test_merge_track_stores_follows_manifest_order_and_uses_fallback(tmp_path):
    rows = []
    for index, name in enumerate(("a.jpg", "b.jpg")):
        target = tmp_path / name; target.write_bytes(b"x")
        clip = tmp_path / f"{name}.mp4"; clip.write_bytes(b"x")
        rows.append({
            "official_split": "train", "partition": "train_core", "file_name": name,
            "target_image_path": str(target), "clip_path": str(clip), "source_video_id": name,
            "duration_seconds": 5.0, "fps": 30.0, "num_frames": 151,
            "target_timestamp_seconds": 5.0, "target_frame_index": 150,
            "action": [1, 0, 0, 0], "reason": [0] * 21,
        })
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    torch.save({"file_names": ["b.jpg"], "tracks_xy": torch.ones(1, 2, 3, 2),
                "visibility": torch.ones(1, 2, 3, dtype=torch.bool)}, first)
    torch.save({"file_names": ["a.jpg"], "tracks_xy": torch.zeros(1, 2, 3, 2),
                "visibility": torch.zeros(1, 2, 3, dtype=torch.bool)}, second)

    payload = merge_track_stores(manifest, [first, second])

    assert payload["file_names"] == ["a.jpg", "b.jpg"]
    assert payload["tracks_xy"].shape == (2, 2, 3, 2)
    assert payload["tracks_xy"][0].sum() == 0
    assert payload["tracks_xy"][1].sum() > 0
    assert payload["audit"]["pass"] is True
