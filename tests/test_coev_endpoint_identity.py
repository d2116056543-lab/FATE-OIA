import json

from fate_oia.engine.audit_coev_endpoint_identity import _write_corrected_manifest


def test_corrected_manifest_preserves_targets_and_only_disables_bad_history(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [
        {"file_name": "good.jpg", "history_available": True, "action": [1, 0, 0, 0]},
        {"file_name": "bad.jpg", "history_available": True, "action": [0, 1, 0, 0]},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "corrected.jsonl"
    _write_corrected_manifest(source, output, {"bad.jpg"})
    corrected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["file_name"] for row in corrected] == ["good.jpg", "bad.jpg"]
    assert corrected[0]["history_available"] is True
    assert corrected[1]["history_available"] is False
    assert corrected[1]["history_unavailable_reason"] == "endpoint_identity_mismatch"
    assert corrected[1]["action"] == rows[1]["action"]
