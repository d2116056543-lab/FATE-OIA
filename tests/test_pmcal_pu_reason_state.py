from __future__ import annotations

import torch


def test_pmcal_pu_state_does_not_make_all_zero_labels_negative():
    from fate_oia.models.pmcal_reason_formula_bank import PMCalReasonFormulaBank
    from fate_oia.models.pmcal_pu_reason_state import PMCalPUReasonState
    names = [f"p{i}" for i in range(32)]
    bank = PMCalReasonFormulaBank("configs/acpr_reason_predicate_grammar.yaml", names)
    builder = PMCalPUReasonState(bank)
    reason = torch.zeros(2, 21)
    q = torch.zeros(2, 32)
    rho = torch.zeros(2, 32)
    state = builder.build(reason, q, rho)
    assert state["reliable_negative_mask"].sum().item() == 0
    assert state["unknown_mask"].sum().item() == 42
