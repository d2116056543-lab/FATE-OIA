from pathlib import Path


def test_head_probe_allows_native_stderr_without_disabling_setup_failures():
    script = (
        Path(__file__).parents[1]
        / "scripts"
        / "FATE_OIA_tida_trajectory_v5_head_probe.ps1"
    ).read_text(encoding="utf-8")

    invoke_at = script.index("& $python")
    continue_at = script.index('$ErrorActionPreference = "Continue"')
    restore_at = script.index('$ErrorActionPreference = "Stop"', invoke_at)

    assert continue_at < invoke_at < restore_at
    assert "$code = $LASTEXITCODE" in script[invoke_at:restore_at]
