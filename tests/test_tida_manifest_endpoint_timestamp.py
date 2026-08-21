import pytest

from fate_oia.engine.build_tida_clip_manifest import _endpoint_timestamp_seconds


def test_manifest_endpoint_timestamp_uses_last_frame_index():
    assert _endpoint_timestamp_seconds(151, 30.0) == pytest.approx(5.0)


def test_manifest_endpoint_timestamp_rejects_invalid_metadata():
    with pytest.raises(ValueError):
        _endpoint_timestamp_seconds(0, 30.0)
