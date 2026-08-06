from pathlib import Path


def test_no_dense_query_patch_value_tensor_contract():
    text = Path("fate_oia/models/aie_evidence_interface.py").read_text(encoding="utf-8")
    assert "expand(b, 4, 4, 3, 3600" not in text
    assert "[B,Q,N,D]" not in text

