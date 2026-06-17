import torch
from fate_oia.models.acpr_predicate_patch_targets import ACPRPredicatePatchTargetBuilder


def test_synthetic_bbox_maps_to_patch_mask():
    builder = ACPRPredicatePatchTargetBuilder(scene_config="configs/acpr_scene_predicates.yaml", grid_hw=(45, 80))
    records = {
        "sample.jpg": {
            "objects": [{"category": "car", "box2d": {"x1": 560, "y1": 420, "x2": 720, "y2": 650}}],
            "lanes": [],
            "drivable": None,
        }
    }
    out = builder.build(["sample.jpg"], records=records)
    assert out["predicate_patch_targets"].shape[-1] == 3600
    assert out["predicate_patch_mask"].sum() > 0
    assert int(out["predicate_source"].max()) >= 0


def test_missing_annotations_do_not_create_high_confidence_masks():
    builder = ACPRPredicatePatchTargetBuilder(scene_config="configs/acpr_scene_predicates.yaml", grid_hw=(45, 80))
    out = builder.build(["missing.jpg"], records={})
    assert out["predicate_patch_targets"].shape == (1, builder.num_predicates, 3600)
    assert float(out["predicate_patch_mask"].sum()) == 0.0
    assert float(out["predicate_patch_reliability"].max()) <= 0.25
