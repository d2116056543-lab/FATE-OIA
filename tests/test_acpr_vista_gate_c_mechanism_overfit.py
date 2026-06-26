from __future__ import annotations

import torch

from fate_oia.engine.audit_acpr_vista_gates import _gate_c


def test_gate_c_runs_real_adapter_mechanism_check():
    payload = _gate_c(torch.device("cpu"), samples=16)
    assert "initial_loss" in payload
    assert "final_loss" in payload
    assert payload["final_loss"] < payload["initial_loss"]
