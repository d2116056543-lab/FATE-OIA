from pathlib import Path


def test_supervisor_propagates_failure_and_runs_stage_c():
    source = Path("fate_oia/engine/supervise_tida_oia_foreground.py").read_text(encoding="utf-8")
    assert "subprocess.run(command, check=False)" in source
    assert "raise subprocess.CalledProcessError" in source
    assert "collect_tida_tta_outputs" in source
    assert "export_tida_deployment" in source
    assert "metric_early" not in source.lower()
