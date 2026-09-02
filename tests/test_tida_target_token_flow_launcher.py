from pathlib import Path


def test_v27_launcher_binds_full_15_frame_target_private_protocol():
    text = Path(
        "scripts/FATE_OIA_tida_target_token_flow_v27_full.ps1"
    ).read_text(encoding="utf-8")
    assert "checkpoint_stage_b_continued.pth" in text
    assert "frame15_coverage_after.json" in text
    assert "missing_available_count" in text
    assert "history_unavailable_fallback_count" in text
    assert '"--batch-size", "6"' in text
    assert '"--gradient-accumulation-steps", "5"' in text
    assert '"--num-workers", "4"' in text
    assert '"--train-owners", "target_token_action,target_token_reason"' in text
    assert '"--run-kind", "full"' in text
    assert "--verified-baseline-artifact" not in text
    invocation = text.index("& $python @arguments")
    assert text.rfind('$ErrorActionPreference = "Continue"', 0, invocation) >= 0
    assert '$ErrorActionPreference = "Stop"' in text[invocation:]
    assert "Start-Process" not in text
    assert "Start-Job" not in text
