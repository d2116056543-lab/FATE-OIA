from __future__ import annotations

import inspect
from pathlib import Path

from fate_oia.engine import audit_acpr_vista_gates
from fate_oia.engine.audit_acpr_vista_gates import _gate_e, _lane_objects_to_mask


def test_gate_e_blocks_without_real_localization_audit(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("bdd100k_root: E:\\sbw\\BDD100K\n", encoding="utf-8")
    payload = _gate_e(str(cfg))
    assert payload["pass"] is False
    assert payload["bdd100k_root_configured"] is True
    assert "localization" in payload["reason"]


def test_gate_e_tracks_object_lane_and_drivable_masks_separately():
    src = inspect.getsource(audit_acpr_vista_gates._gate_e)
    assert "object_delta_mass_mean" in src
    assert "lane_delta_mass_mean" in src
    assert "drivable_delta_mass_mean" in src
    mask_src = inspect.getsource(audit_acpr_vista_gates._masks_for_file)
    assert "include_lane=False" in mask_src
    assert "cat.startswith(\"lane/\")" in mask_src
    assert "drivable_map_to_mask" in mask_src


def test_gate_e_rasterizes_bdd100k_lane_polyline_vertices():
    lane_obj = {
        "category": "lane/single white",
        "poly2d": [[917.056349, 391.597047, "L"], [1087.880571, 422.56518, "L"]],
    }
    mask = _lane_objects_to_mask([lane_obj], image_size=(1280, 720), output_size=(45, 80))
    assert float(mask.sum()) > 0.0
