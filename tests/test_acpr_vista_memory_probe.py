from __future__ import annotations

from pathlib import Path


def test_memory_probe_module_exists():
    assert Path("fate_oia/engine/probe_acpr_vista_memory.py").exists()

