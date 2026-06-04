from fate_oia.datasets.bdd100k_scene_state import scene_state_from_bdd100k_record


def test_scene_state_uses_box_geometry_for_front_left_right_counts():
    record = {
        "frames": [{
            "objects": [
                {"category": "car", "box2d": {"x1": 280, "y1": 330, "x2": 360, "y2": 430}},
                {"category": "person", "box2d": {"x1": 20, "y1": 300, "x2": 90, "y2": 430}},
                {"category": "traffic light", "box2d": {"x1": 500, "y1": 40, "x2": 540, "y2": 100}},
            ]
        }],
        "width": 640,
        "height": 480,
    }
    out = scene_state_from_bdd100k_record(record)
    counts = out["counts"]
    assert counts["front_vehicle_count"] >= 1
    assert counts["left_object_count"] >= 1
    assert counts["right_object_count"] >= 1
    assert counts["traffic_control_count"] >= 1


def test_scene_state_uses_lane_poly_geometry_and_drivable_weak_flag():
    record = {
        "frames": [{
            "objects": [
                {"category": "lane/single white", "poly2d": [{"vertices": [[80, 300], [120, 460]], "types": ["L", "L"]}]},
                {"category": "lane/single yellow", "poly2d": [{"vertices": [[500, 300], [540, 460]], "types": ["L", "L"]}]},
            ]
        }],
        "_drivable_map": "dummy.png",
        "width": 640,
        "height": 480,
    }
    out = scene_state_from_bdd100k_record(record)
    counts = out["counts"]
    assert counts["lane_left_count"] >= 1
    assert counts["lane_right_count"] >= 1
    assert out["proxy_quality"]["direct_drivable_proxy"] in {"weak_drivable_map_presence", "drivable_map_read"}
