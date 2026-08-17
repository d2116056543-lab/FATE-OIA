from pathlib import Path


def test_calibration_is_fit_before_and_only_from_calib_logits():
    text = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    section = text[text.index("def evaluate_epoch"):text.index("def main")]
    assert section.index("collect_logits(model, calib_loader") < section.index("collect_logits(model, test_loader")
    assert "fit_posthoc_thresholds" in section and "test[" not in section.split("fit_posthoc_thresholds", 1)[0]


def test_clean_checkpoint_selection_is_computed_from_train_audit_not_test_metrics():
    text = Path("fate_oia/engine/train_aie_oia.py").read_text(encoding="utf-8")
    loop_start = text.index("for epoch in range(start_epoch, epochs)")
    loop_end = text.index("if args.run_kind == \"full\":", loop_start)
    epoch_loop = text[loop_start:loop_end]

    assert "selection_metrics = evaluate_selection_epoch" in epoch_loop
    assert "criteria = checkpoint_selection_criteria(selection_metrics)" in epoch_loop
    assert "checkpoint_best_train_audit_" in epoch_loop
    assert "checkpoint_best_test_" not in epoch_loop

