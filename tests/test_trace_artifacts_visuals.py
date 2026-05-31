from fate_oia.engine.export_trace_visuals import export_trace_case

def test_visual_artifact_schema(tmp_path):
    row = export_trace_case(tmp_path, 0, {"sample_id": "x", "reason_idx": 1, "prototype_id": 2, "drop": 0.1, "top_evidence": [{"source_type": "object", "transport_mass": 1.0}]})
    assert (tmp_path / "visuals" / "epoch_000" / "x.png").exists()
    assert row["top_evidence"][0]["source_type"] == "object"
