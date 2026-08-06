from pathlib import Path


def test_standalone_eval_uses_checkpoint_canonicalization():
    source = Path("fate_oia/engine/eval_aie_oia.py").read_text(encoding="utf-8")
    assert "canonical_model_state_dict(checkpoint[\"model\"])" in source
