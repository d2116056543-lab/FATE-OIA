from pathlib import Path


def test_pilot_gate_reads_raw_artifacts_and_does_not_embed_pass():
    text = Path("fate_oia/engine/evaluate_aie_oia_pilot.py").read_text(encoding="utf-8")
    assert "read_jsonl" in text and "metrics_summary.jsonl" in text and "evidence_components.jsonl" in text
    assert "passed = all(gates.values())" in text

