from pathlib import Path

from fate_oia.models.meter_semantic_action import StateConditionedActionCredit


def test_heca_active_path_has_no_sample_admission_gate() -> None:
    module = StateConditionedActionCredit(dim=8, factor_dim=3, rank=2)
    assert not any("admission" in name for name, _ in module.named_parameters())
    active = [
        Path("fate_oia/models/meter_semantic_action.py"),
        Path("fate_oia/models/meter_oia_model.py"),
        Path("fate_oia/engine/train_acpr_meter_oia.py"),
        Path("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in active)
    assert "action_evidence_admission" not in text

