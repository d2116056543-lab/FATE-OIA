from pathlib import Path


def test_counterfactual_engine_never_calls_image_encoder():
    text = Path("fate_oia/utils/aie_counterfactual.py").read_text(encoding="utf-8")
    assert ".encode_images(" not in text and ".dino(" not in text
    assert "rerun_action_evidence_from_field" in text
    assert "wrong_probe_drop" in text and "wrong_action_drop" in text
