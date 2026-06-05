from pathlib import Path
from fate_oia.datasets.bdd100k_scene_state_proxy import build_scene_state_proxy


def test_reason_names_external_note_and_no_test_gt_forward():
    text = Path("configs/egcaf_factor_groups.yaml").read_text(encoding="utf-8")
    assert "Raw BDD-OIA JSON stores 21 index positions" in text
    model_text = Path("fate_oia/models/egcaf_oia_model.py").read_text(encoding="utf-8")
    assert "bdd100k_scene_state" in model_text
    assert "BDD100K" not in model_text
