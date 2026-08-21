from fate_oia.utils.tida_contracts import validate_git_binding


def test_git_binding_requires_all_heads_to_match():
    assert validate_git_binding("abc", "abc", "abc") == []
    assert validate_git_binding("abc", "def", "abc")
