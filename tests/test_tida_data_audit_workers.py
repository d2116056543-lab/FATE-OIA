import pytest

from fate_oia.engine.audit_tida_video_data import audit_manifest


def test_data_audit_rejects_nonpositive_worker_count(tmp_path):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        audit_manifest(manifest, tmp_path / "audit", workers=0)
