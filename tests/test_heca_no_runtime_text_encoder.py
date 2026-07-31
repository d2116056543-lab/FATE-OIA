from pathlib import Path


def test_runtime_model_loads_static_prototypes_without_text_encoder() -> None:
    source = Path("fate_oia/models/meter_signed_factors.py").read_text(encoding="utf-8")
    assert "CLIPTextModel" not in source
    assert "BertModel" not in source
    assert "AutoTokenizer" not in source
    assert "factor_text_prototype" in source

