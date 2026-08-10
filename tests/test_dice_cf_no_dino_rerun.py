from pathlib import Path


def test_cf_engine_only_calls_dice_rerun():
    source=Path("fate_oia/utils/dice_counterfactual_engine.py").read_text(encoding="utf-8")
    assert "rerun_dice_from_conditioned" in source
    assert "encode_images" not in source and ".dino(" not in source
