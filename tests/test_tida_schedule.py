from fate_oia.utils.tida_contracts import schedule_values


def test_temporal_scale_uses_one_continuous_update_schedule():
    assert schedule_values(0, 100)["temporal_scale"] == 0.0
    assert schedule_values(5, 100)["temporal_scale"] == 0.0
    assert 0.0 < schedule_values(10, 100)["temporal_scale"] < 1.0
    assert schedule_values(20, 100)["temporal_scale"] == 1.0
