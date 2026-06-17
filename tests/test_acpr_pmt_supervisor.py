from pathlib import Path


def test_pmt_supervisor_foreground_script_has_no_background_processes():
    text = Path("scripts/FATE_OIA_acpr_pmt_s_v1_foreground.ps1").read_text(encoding="utf-8")
    forbidden = ["Start-Process", "Start-Job", "nohup", "scheduled task"]
    assert not any(x in text for x in forbidden)
    assert "supervise_acpr_pmt_s_foreground" in text


def test_pmt_supervisor_runs_full_preflight_before_full_train():
    text = Path("fate_oia/engine/supervise_acpr_pmt_s_foreground.py").read_text(encoding="utf-8")
    for required in ["py_compile", "pytest", "audit_acpr_pmt_s_implementation", "REVIEW_PASS_ACPR_PMT_S_V1", "max_train_samples", "git", "ls-remote"]:
        assert required in text
