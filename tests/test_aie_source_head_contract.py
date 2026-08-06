from pathlib import Path


def test_source_head_is_locked_in_audit_and_manifest_code():
    expected = "373aa49feac17372574fd7fb056c1d79c7c848fe"
    assert expected in Path("fate_oia/engine/audit_aie_oia_implementation.py").read_text(encoding="utf-8")
    assert expected in Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")

