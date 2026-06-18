from pathlib import Path


def test_seca_supervisor_foreground_only():
    text = Path("scripts/FATE_OIA_acpr_seca_v1_foreground.ps1").read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "Start-Job" not in text
    assert "fate_oia.engine.supervise_acpr_seca_foreground" in text



def test_seca_supervisor_runs_preflight_and_fallback():
    text = Path("fate_oia/engine/supervise_acpr_seca_foreground.py").read_text(encoding="utf-8")
    assert "audit_acpr_seca_implementation" in text
    assert "acpr_seca_v1_supervisor_smoke" in text
    assert "FALLBACKS" in text
    assert "out of memory" in text.lower()
