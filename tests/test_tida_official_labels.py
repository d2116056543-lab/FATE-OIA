import json

import pytest

from fate_oia.datasets.tida_official_labels import (
    load_action_label_map,
    load_official_label_map,
    load_reason_label_map,
)


def test_coco_action_labels_join_images_to_annotations(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text(json.dumps({
        "images": [
            {"id": 7, "file_name": "a.jpg"},
            {"id": 9, "file_name": "b.jpg"},
        ],
        "annotations": [
            {"id": 9, "category": [0, 1, 0, 1, 0]},
            {"id": 7, "category": [1, 0, 1, 0, 1]},
        ],
    }), encoding="utf-8")

    labels = load_action_label_map(path)

    assert labels == {"a.jpg": (1.0, 0.0, 1.0, 0.0), "b.jpg": (0.0, 1.0, 0.0, 1.0)}


def test_reason_labels_use_file_name_and_require_21_values(tmp_path):
    path = tmp_path / "reasons.json"
    values = [0] * 21
    values[4] = 1
    path.write_text(json.dumps([{"file_name": "a.jpg", "reason": values}]), encoding="utf-8")

    assert load_reason_label_map(path)["a.jpg"][4] == 1.0

    path.write_text(json.dumps([{"file_name": "a.jpg", "reason": values[:-1]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="21"):
        load_reason_label_map(path)


def test_official_pair_rejects_missing_or_ambiguous_labels(tmp_path):
    action = tmp_path / "actions.json"
    reason = tmp_path / "reasons.json"
    action.write_text(json.dumps({
        "images": [{"id": 0, "file_name": "a.jpg"}],
        "annotations": [{"id": 0, "category": [1, 0, 0, 0, 0]}],
    }), encoding="utf-8")
    reason.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(ValueError, match="label key mismatch"):
        load_official_label_map(action, reason)

    action.write_text(json.dumps({
        "images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "a.jpg"}],
        "annotations": [
            {"id": 0, "category": [1, 0, 0, 0, 0]},
            {"id": 1, "category": [0, 1, 0, 0, 0]},
        ],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_action_label_map(action)
