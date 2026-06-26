from __future__ import annotations

from pathlib import Path


def test_faithfulness_module_exists():
    assert Path("fate_oia/engine/eval_acpr_vista_faithfulness.py").exists()

