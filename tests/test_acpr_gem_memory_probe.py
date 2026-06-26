from fate_oia.engine.probe_acpr_gem_memory import candidate_pairs


def test_memory_probe_uses_required_candidate_ladder():
    assert candidate_pairs(["6:5", "5:6", "4:8", "3:10", "2:15"]) == [(6, 5), (5, 6), (4, 8), (3, 10), (2, 15)]
