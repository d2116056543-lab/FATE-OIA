from pathlib import Path


def test_trainer_uses_grounded_naming_alignment_loss():
    source = (Path(__file__).parents[1] / "fate_oia" / "engine" / "train_aie_oia.py").read_text()
    assert "naming_alignment_loss(" in source
    assert 'output["name_spatial_soft_iou"][sample' not in source
