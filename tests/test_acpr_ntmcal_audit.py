import subprocess, sys

def test_audit_module_help():
    r = subprocess.run([sys.executable, "-m", "fate_oia.engine.audit_acpr_ntmcal_implementation", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
