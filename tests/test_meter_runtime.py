from fate_oia.utils.meter_runtime import METERRuntimeProfile, choose_meter_profile


def test_runtime_profile_rejects_hard_memory_limit() -> None:
    result = choose_meter_profile([METERRuntimeProfile(8, 4, 40.0, 10.0), METERRuntimeProfile(16, 2, 46.0, 100.0)])
    assert result.reserved_gb < 45.0
