from __future__ import annotations

import json
from pathlib import Path

from fate_oia.datasets.bdd100k_structured import BDD100KStructuredIndex, bdd_oia_to_bdd100k_stem


def test_stem_strip_suffix() -> None:
    assert bdd_oia_to_bdd100k_stem("abc_1.jpg") == "abc"
    assert bdd_oia_to_bdd100k_stem("abc-def.jpg") == "abc-def"


def test_parser_reads_boxes_poly_and_drivable(tmp_path: Path) -> None:
    label_dir = tmp_path / "bdd100k_labels" / "bdd100k" / "labels" / "100k" / "val"
    drive_dir = tmp_path / "bdd100k_drivable_maps" / "bdd100k" / "drivable_maps" / "color_labels" / "val"
    label_dir.mkdir(parents=True)
    drive_dir.mkdir(parents=True)
    (label_dir / "sample.json").write_text(
        json.dumps(
            {
                "attributes": {"weather": "clear"},
                "frames": [
                    {
                        "objects": [
                            {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 20}},
                            {"category": "lane/crosswalk", "poly2d": [{"vertices": [[1, 1], [2, 2], [3, 1]], "closed": True}]},
                            {"category": "area/drivable", "poly2d": [[1, 1, "L"], [4, 1, "L"], [4, 4, "L"]]},
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (drive_dir / "sample_drivable_color.png").write_bytes(b"x")
    idx = BDD100KStructuredIndex(tmp_path)
    rec = idx.lookup("sample_1.jpg", "test")
    assert rec.label_path is not None
    assert rec.box_count == 1
    assert len(rec.lanes) == 2
    assert rec.has_drivable
    audit = idx.audit_samples(["sample_1.jpg"], "test")
    assert audit["matched_count"] == 1
    assert audit["lane_count"] == 2
