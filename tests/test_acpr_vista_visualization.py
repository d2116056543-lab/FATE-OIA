from __future__ import annotations

from pathlib import Path


def test_visual_export_module_exists():
    assert Path("fate_oia/engine/export_acpr_vista_visuals.py").exists()

