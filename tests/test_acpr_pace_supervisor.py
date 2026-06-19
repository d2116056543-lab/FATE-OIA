from pathlib import Path


def test_supervisor_contains_required_gates_and_no_detach():
    text = Path("fate_oia/engine/supervise_acpr_pace_foreground.py").read_text(encoding="utf-8")
    for required in ["git ls-remote", "REVIEW_PASS_ACPR_PACE_V1.txt", "audit_acpr_pace_signal", "heartbeat", "fallback_ladder"]:
        assert required in text
    for forbidden in ["Start-Process", "Start-Job", "nohup", "scheduled task"]:
        assert forbidden not in text
