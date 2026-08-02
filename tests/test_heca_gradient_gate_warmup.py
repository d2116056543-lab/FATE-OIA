from fate_oia.engine.evaluate_meter_oia_v3_heca_pilot import _ownership_gate_rows


def _row(*, ramp: float, ratio: float, state_effect_norm: float = 0.12) -> dict[str, float]:
    return {
        "action_to_anchor_query": 0.0,
        "action_to_state_bridge_ratio": ratio,
        "action_to_credit_adapter": 0.01,
        "reason_to_action_credit": 0.0,
        "pu_to_action_factor": 0.0,
        "measurement_to_foundation": 0.0,
        "action_credit_ramp": ramp,
        "action_state_effect_norm": state_effect_norm,
    }


def test_ownership_gate_ignores_zero_init_rows_before_credit_is_active() -> None:
    rows = [_row(ramp=0.0, ratio=0.0), _row(ramp=1.0, ratio=0.02)]

    eligible, checks = _ownership_gate_rows(rows)

    assert len(eligible) == 1
    assert checks == [True]


def test_ownership_gate_rejects_active_state_bridge_below_floor() -> None:
    eligible, checks = _ownership_gate_rows([_row(ramp=1.0, ratio=0.003)])

    assert len(eligible) == 1
    assert checks == [False]


def test_ownership_gate_ignores_full_ramp_rows_before_state_effect_matures() -> None:
    rows = [
        _row(ramp=1.0, ratio=0.003, state_effect_norm=0.02),
        _row(ramp=1.0, ratio=0.02, state_effect_norm=0.12),
    ]

    eligible, checks = _ownership_gate_rows(rows)

    assert len(eligible) == 1
    assert checks == [True]


def test_ownership_gate_fails_closed_when_active_ramp_telemetry_is_missing() -> None:
    row = _row(ramp=1.0, ratio=0.02)
    row.pop("action_credit_ramp")

    eligible, checks = _ownership_gate_rows([row])

    assert eligible == []
    assert checks == [False]
