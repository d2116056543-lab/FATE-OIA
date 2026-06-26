from __future__ import annotations

from fate_oia.engine.audit_acpr_vista_gates import _gate_d


def test_gate_d_blocks_when_train_calib_sanity_not_executed():
    payload = _gate_d("")
    assert payload["pass"] is False
    assert payload["required_before_full_train"] is True
    assert "train_calib" in payload["reason"]
