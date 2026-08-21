from fate_oia.utils.tida_contracts import choose_memory_candidate


def test_memory_probe_selects_fastest_safe_candidate():
    rows = [
        {"name": "A", "peak_reserved_gib": 46.0, "growth_gib": 0.0, "samples_per_second": 10.0},
        {"name": "B", "peak_reserved_gib": 44.0, "growth_gib": 0.1, "samples_per_second": 8.0},
        {"name": "C", "peak_reserved_gib": 40.0, "growth_gib": 0.1, "samples_per_second": 8.1},
    ]
    assert choose_memory_candidate(rows)["name"] == "C"
