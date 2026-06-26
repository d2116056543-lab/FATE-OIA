import json
from pathlib import Path

import torch
from PIL import Image

from fate_oia.grounding.acpr_gem_grounding import ACPRGEMGroundingBuilder


def test_grounding_builder_rasterizes_object_lane_and_drivable(tmp_path: Path):
    root = tmp_path / "BDD100K"
    label_dir = root / "bdd100k_labels" / "100k" / "train"
    drive_dir = root / "bdd100k_drivable_maps" / "100k" / "train"
    label_dir.mkdir(parents=True)
    drive_dir.mkdir(parents=True)
    label = {
        "frames": [
            {
                "objects": [
                    {"category": "car", "box2d": {"x1": 540, "y1": 430, "x2": 740, "y2": 620}},
                    {"category": "lane/single white", "poly2d": [[917.0, 391.0, "L"], [1087.0, 422.0, "L"]]},
                    {"category": "area/drivable", "poly2d": [[400, 500, "L"], [900, 500, "L"], [900, 719, "L"], [400, 719, "L"]]},
                ]
            }
        ]
    }
    (label_dir / "abc.json").write_text(json.dumps(label), encoding="utf-8")
    img = Image.new("L", (1280, 720), 0)
    img.paste(76, (400, 500, 900, 719))
    img.save(drive_dir / "abc_drivable_id.png")

    builder = ACPRGEMGroundingBuilder(root, "configs/acpr_gem_evidence_slots.yaml")
    out = builder.build(["abc_1.jpg"], device=torch.device("cpu"))

    assert out["grounding_targets"].shape == (1, 20, 3600)
    assert out["grounding_mask"].sum() >= 3
    assert out["grounding_stats"]["object_slot_available"] > 0
    assert out["grounding_stats"]["lane_slot_available"] > 0
    assert out["grounding_stats"]["drivable_slot_available"] > 0
