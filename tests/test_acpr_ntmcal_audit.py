import subprocess
import sys
from pathlib import Path


def test_audit_module_help():
    r = subprocess.run(
        [sys.executable, "-m", "fate_oia.engine.audit_acpr_ntmcal_implementation", "--help"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0


def test_audit_declares_component_level_functional_checks():
    text = Path("fate_oia/engine/audit_acpr_ntmcal_implementation.py").read_text(encoding="utf-8")
    required = [
        "dataset_direct_image",
        "dino_frozen",
        "native_text_bank",
        "text_atom_encoder",
        "reason_formulas",
        "topk_measurement",
        "observation_builder",
        "predicate_loss",
        "pu_state",
        "reason_residual",
        "action_predicate",
        "threshold_head",
        "pair_memory",
        "full_model_forward",
        "training_protocol",
        "artifact_schema",
        "memory_probe",
        "foreground_supervisor",
    ]
    for name in required:
        assert f'"{name}"' in text
