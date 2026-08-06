from pathlib import Path


def test_runtime_profile_can_recheck_one_declared_candidate():
    source = Path("fate_oia/engine/profile_aie_oia.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--candidate"' in source
    assert "selected_candidate not in CANDIDATES" in source
    assert "candidates = (selected_candidate,)" in source
