import pytest

from fate_oia.datasets.tida_clip_manifest import assert_no_partition_leakage


def test_source_or_content_collision_fails():
    rows = [
        {"partition": "train_core", "source_video_id": "same", "clip_sha256": "a", "endpoint_phash": 1},
        {"partition": "test", "source_video_id": "same", "clip_sha256": "b", "endpoint_phash": 10},
    ]
    with pytest.raises(ValueError, match="leakage"):
        assert_no_partition_leakage(rows)
