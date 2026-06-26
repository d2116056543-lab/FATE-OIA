from pathlib import Path

from fate_oia.engine.audit_acpr_gem_gates import gate_file_names


def test_gate_file_contract_contains_all_blocking_gates():
    names = gate_file_names()
    for name in [
        "GEM_GATE_A_EQUIVALENCE.json",
        "GEM_GATE_B_ORACLE_UPPER_BOUND.json",
        "GEM_GATE_C_LEARNED_GROUNDING.json",
        "GEM_GATE_D_MECHANISM_OVERFIT.json",
        "GEM_GATE_E_TRAIN_CALIB_SANITY.json",
        "GEM_GATE_F_FAITHFULNESS.json",
        "GEM_MEMORY_PASS.json",
        "GEM_GATES_PASS.json",
    ]:
        assert name in names
