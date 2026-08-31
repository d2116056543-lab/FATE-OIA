from types import SimpleNamespace

from fate_oia.engine.extract_tida_raw_frames import select_extractable_records


def test_full_frame_extraction_uses_every_available_manifest_record():
    rows = [
        SimpleNamespace(file_name="a.jpg", history_available=True),
        SimpleNamespace(file_name="b.jpg", history_available=False),
        SimpleNamespace(file_name="c.jpg", history_available=True),
    ]
    assert [row.file_name for row in select_extractable_records(rows)] == ["a.jpg", "c.jpg"]


def test_track_subset_remains_supported_without_extracting_invalid_history():
    rows = [
        SimpleNamespace(file_name="a.jpg", history_available=True),
        SimpleNamespace(file_name="b.jpg", history_available=False),
        SimpleNamespace(file_name="c.jpg", history_available=True),
    ]
    assert [
        row.file_name for row in select_extractable_records(rows, {"A.JPG", "b.jpg"})
    ] == ["a.jpg"]
