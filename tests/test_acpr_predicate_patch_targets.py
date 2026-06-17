from pathlib import Path

from fate_oia.models.acpr_predicate_patch_targets import ACPRPredicatePatchTargetBuilder, SOURCE_IDS


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


def test_bdd100k_root_loads_json_and_drivable_without_explicit_records():
    root = Path(r"E:\sbw\BDD100K")
    if not root.exists():
        return
    builder = ACPRPredicatePatchTargetBuilder(
        scene_config="configs/acpr_scene_predicates.yaml",
        bdd100k_root=root,
        grid_hw=(45, 80),
    )
    out = builder.build(["0000f77c-6257be58.jpg"])
    coverage = out["predicate_patch_coverage"]
    assert float(out["predicate_patch_mask"].sum()) > 0.0
    assert coverage["object_box_count"] > 0
    assert coverage["lane_poly_count"] > 0
    assert coverage["drivable_count"] > 0
    assert int((out["predicate_source"] == SOURCE_IDS["weak_region"]).sum()) >= 0


def test_bdd_oia_frame_suffix_maps_to_bdd100k_annotation_stem():
    root = Path(r"E:\sbw\BDD100K")
    if not root.exists():
        return
    builder = ACPRPredicatePatchTargetBuilder(
        scene_config="configs/acpr_scene_predicates.yaml",
        bdd100k_root=root,
        grid_hw=(45, 80),
    )
    out = builder.build(["9af44b54-7aae6c7d_3.jpg"])
    assert float(out["predicate_patch_mask"].sum()) > 0.0
    assert out["predicate_patch_coverage"]["object_box_count"] > 0


def test_missing_annotations_do_not_create_high_confidence_masks():
    builder = ACPRPredicatePatchTargetBuilder(scene_config="configs/acpr_scene_predicates.yaml", grid_hw=(45, 80))
    out = builder.build(["missing.jpg"], records={})
    assert out["predicate_patch_targets"].shape == (1, builder.num_predicates, 3600)
    assert float(out["predicate_patch_mask"].sum()) == 0.0
    assert float(out["predicate_patch_reliability"].max()) <= 0.25
