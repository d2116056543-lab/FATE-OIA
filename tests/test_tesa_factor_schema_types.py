from pathlib import Path

import yaml


def test_tesa_factor_schema_is_complete() -> None:
    rows = yaml.safe_load(Path("configs/meter_factor_schema.yaml").read_text())["factors"]
    required = {
        "id", "name", "factor_type", "state_set", "anchor_source",
        "state_source", "groundability", "action_owned",
        "observability_source", "mirror_partner", "counter_localizable",
    }
    assert [row["id"] for row in rows] == list(range(21))
    assert all(required <= row.keys() for row in rows)
    assert all(len(row["state_set"]) >= 3 and row["state_set"][-1] == "unknown" for row in rows)
    assert rows[14]["action_owned"] == rows[20]["action_owned"] == 0
    assert rows[1]["action_owned"] == 0.5
    assert all(row["counter_localizable"] is False for row in rows)
    for left, right in ((9, 15), (10, 16), (11, 17), (12, 18), (13, 19), (14, 20)):
        assert rows[left]["mirror_partner"] == right
        assert rows[right]["mirror_partner"] == left
