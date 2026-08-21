import pytest

from fate_oia.engine.audit_tida_video_data import (
    _load_complete_sample_artifact,
    audit_manifest,
    exclusive_audit_lock,
)


def test_data_audit_rejects_nonpositive_worker_count(tmp_path):
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="workers"):
        audit_manifest(manifest, tmp_path / "audit", workers=0)


def test_data_audit_lock_rejects_a_second_writer(tmp_path):
    output = tmp_path / "audit"
    with exclusive_audit_lock(output):
        with pytest.raises(RuntimeError, match="another TIDA data audit"):
            with exclusive_audit_lock(output):
                pass
    assert not (output / ".tida_data_audit.lock").exists()


def test_complete_artifact_reuse_rejects_missing_prior_files(tmp_path):
    with pytest.raises(RuntimeError, match="complete prior"):
        _load_complete_sample_artifact(tmp_path / "missing.jsonl", [], "abc")
