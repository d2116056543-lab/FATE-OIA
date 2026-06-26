from __future__ import annotations

import inspect

from fate_oia.engine import audit_acpr_vista_gates
from fate_oia.engine.audit_acpr_vista_gates import _gate_d


def test_gate_d_blocks_when_train_calib_sanity_not_executed():
    payload = _gate_d("")
    assert payload["pass"] is False
    assert payload["required_before_full_train"] is True
    assert "train_calib" in payload["reason"]


def test_gate_d_can_train_threshold_path_through_final_logits():
    src = inspect.getsource(audit_acpr_vista_gates._task_loss)
    assert "use_final_logits" in src
    assert "action_logits_final_raw" in src
    assert "reason_logits_final_raw" in src
    train_src = inspect.getsource(audit_acpr_vista_gates._train_steps)
    assert "use_final_logits" in train_src
    gate_d_src = inspect.getsource(audit_acpr_vista_gates._gate_d)
    assert "use_final_logits=True" in gate_d_src


def test_gate_d_preserves_action_thresholds_when_refreshing_reason_thresholds():
    gate_d_src = inspect.getsource(audit_acpr_vista_gates._gate_d)
    assert "action_threshold_preserved" in gate_d_src
    assert "teacher_theta[: model.action_dim] = current_theta[: model.action_dim]" in gate_d_src
