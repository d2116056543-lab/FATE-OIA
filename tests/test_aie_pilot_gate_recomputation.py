from pathlib import Path

from fate_oia.engine.evaluate_aie_oia_pilot import probe_health_gate


def test_pilot_gate_reads_raw_artifacts_and_does_not_embed_pass():
    text = Path("fate_oia/engine/evaluate_aie_oia_pilot.py").read_text(encoding="utf-8")
    assert "read_jsonl" in text and "metrics_summary.jsonl" in text and "evidence_components.jsonl" in text
    assert "passed = all(gates.values())" in text


def test_probe_health_uses_late_epochs_not_initialization():
    epochs = [{"epoch": index} for index in range(4)]
    initial_duplicate = {"epoch": 0, "dominant_probe_over_0p9_rate": 0.0, "probe_pairwise_overlap": 0.999,
                         "probe_effective_count": 4.0, "probe_map_entropy": 8.0}
    late_healthy = {"epoch": 3, "dominant_probe_over_0p9_rate": 0.1, "probe_pairwise_overlap": 0.7,
                    "probe_effective_count": 2.5, "probe_map_entropy": 7.0}
    assert probe_health_gate([initial_duplicate, late_healthy], epochs)
    assert not probe_health_gate([initial_duplicate], epochs)
