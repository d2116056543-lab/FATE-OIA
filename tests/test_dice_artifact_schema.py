from fate_oia.utils.dice_artifacts import REQUIRED_GATE_FILES
from pathlib import Path


def test_gate_schema_contains_all_binding_and_decision_artifacts():
    assert set(REQUIRED_GATE_FILES)=={"DICE_BASE_REPLAY.json","DICE_IMPLEMENTATION_REVIEW.json","DICE_ORACLE_POTENTIAL.json","DICE_PROBE_METRICS.json","DICE_PAIRED_BOOTSTRAP.json","DICE_MECHANISM_GATES.json"}


def test_batch_artifact_source_contains_required_nonplaceholder_diagnostics():
    source=Path("fate_oia/engine/train_dice_oia_probe.py").read_text(encoding="utf-8")
    required=("base_action_logit_rms","dice_action_logit_rms","dice_delta_p10","per_action_delta_mean",
              "license_prediction_auc","certificate_positive_rate","rank_protect","base_pair_inversion_rate",
              "dice_pair_inversion_rate","base_parameter_delta_max","dice_grad_norm","dino_grad",
              "allocated_gb","reserved_gb","data_time","dino_time","dice_time","cf_time","backward_time")
    assert all(name in source for name in required)
