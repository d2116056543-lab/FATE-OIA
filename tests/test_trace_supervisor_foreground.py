from pathlib import Path

def test_supervisor_rejects_background_terms():
    text = Path("fate_oia/engine/supervise_trace_oia_foreground.py").read_text() + Path("scripts/FATE_OIA_trace_oia_v1_foreground.ps1").read_text()
    for term in ["Start-Process", "Start-Job", "Win32_Process", "Invoke-WmiMethod", "nohup"]:
        assert term not in text
    assert "subprocess.Popen" in text and "proc.wait" in text
