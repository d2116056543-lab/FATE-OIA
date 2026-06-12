from pathlib import Path

from fate_oia.engine.audit_cast_oia_implementation import run_audit


def test_audit_reports_missing_review_pass_without_write(tmp_path):
    result = run_audit(
        repo_root=Path("."),
        config=Path("configs/fate_oia_train_360x640_cast_oia_v1.yaml"),
        output_dir=tmp_path,
        write_review_pass=False,
        run_smoke=False,
    )
    assert "functional_checks" in result
    assert "forbidden_patterns" in result
    assert not (tmp_path / "REVIEW_PASS_CAST_OIA_V1.txt").exists()
