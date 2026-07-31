from pathlib import Path


def test_heca_import_graph_has_no_legacy_mask_or_admission() -> None:
    paths = [
        Path("fate_oia/models/meter_schema.py"), Path("fate_oia/models/meter_semantic_action.py"),
        Path("fate_oia/models/meter_oia_model.py"), Path("fate_oia/engine/train_acpr_meter_oia.py"),
        Path("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "compatible_actions" not in source
    assert "action_evidence_admission" not in source
    assert "near_boundary" not in source
    assert "anti_monopoly" not in source

