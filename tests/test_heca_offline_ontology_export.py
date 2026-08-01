from pathlib import Path

import yaml


def test_ontology_export_is_strictly_offline_and_uses_the_project_bert_path() -> None:
    source = Path(
        "fate_oia/engine/export_heca_ontology_prototypes.py"
    ).read_text(encoding="utf-8")
    assert "all-MiniLM-L6-v2" not in source
    assert "frozen_bert_base_uncased" in source
    assert "local_files_only=True" in source
    assert 'os.environ.setdefault("USE_TF", "0")' in source
    assert 'os.environ.setdefault("USE_TORCH", "1")' in source


def test_pilot_passes_the_configured_offline_text_encoder_to_the_exporter() -> None:
    script = Path("scripts/FATE_OIA_meter_oia_v3_heca_pilot.ps1").read_text(
        encoding="utf-8"
    )
    assert "[string]$TextEncoderPath" not in script
    assert '$TextEncoderPath = "artifacts\\heca\\frozen_bert_base_uncased"' in script
    assert '"--encoder_id", $TextEncoderPath' in script
    config = yaml.safe_load(
        Path("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["artifacts"]["offline_text_encoder_path"].endswith(
        "frozen_bert_base_uncased"
    )
