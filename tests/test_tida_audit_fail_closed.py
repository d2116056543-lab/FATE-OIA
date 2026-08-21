from pathlib import Path

from fate_oia.engine.audit_tida_oia_implementation import REQUIRED_FILES


def test_audit_requires_every_formal_file_and_all_four_reviews():
    assert all(Path(path).is_file() for path in REQUIRED_FILES)
    source = Path("fate_oia/engine/audit_tida_oia_implementation.py").read_text(encoding="utf-8")
    for name in ("DESIGN_REVIEW_PASS", "IMPLEMENTATION_REVIEW_PASS", "MECHANISM_REVIEW_PASS", "MEMORY_REVIEW_PASS", "FULL_TRAIN_READY"):
        assert name in source
    assert "remote_matches" in source and "git[\"clean\"]" in source
    assert 'parser.add_argument("--golden-oracle", required=True)' in source
    assert "source_tree_image_oracle_audit.json" in source
