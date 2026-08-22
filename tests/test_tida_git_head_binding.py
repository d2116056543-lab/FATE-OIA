import json
from datetime import datetime, timezone

from fate_oia.engine.audit_tida_oia_implementation import _validate_remote_head_proof
from fate_oia.utils.tida_contracts import validate_git_binding


def test_git_binding_requires_all_heads_to_match():
    assert validate_git_binding("abc", "abc", "abc") == []
    assert validate_git_binding("abc", "def", "abc")


def test_recent_github_remote_proof_must_match_branch_and_head(tmp_path):
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({
        "remote_url": "https://github.com/d2116056543-lab/FATE-OIA.git",
        "branch": "feature",
        "remote_head": "abc",
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }), encoding="utf-8")
    result = _validate_remote_head_proof(proof, head="abc", branch="feature")
    assert result["valid"]
    assert not _validate_remote_head_proof(proof, head="def", branch="feature")["valid"]
