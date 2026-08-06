from pathlib import Path


def test_calibration_is_fit_before_and_only_from_calib_logits():
    text = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    section = text[text.index("def evaluate_epoch"):text.index("def main")]
    assert section.index("collect_logits(model, calib_loader") < section.index("collect_logits(model, test_loader")
    assert "fit_posthoc_thresholds" in section and "test[" not in section.split("fit_posthoc_thresholds", 1)[0]

