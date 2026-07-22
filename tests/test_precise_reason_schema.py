from pathlib import Path

import pytest

from fate_oia.utils.precise_schema import load_action_semantics, load_reason_semantics


ROOT = Path(__file__).resolve().parents[1]


def test_reason_semantics_has_exactly_21_external_named_rows():
    rows = load_reason_semantics(ROOT / "configs" / "precise_reason_semantics.yaml")
    assert len(rows) == 21
    assert [row["id"] for row in rows] == list(range(21))
    for row in rows:
        assert set(("entity", "state", "sector", "decision_role", "allowed_evidence_families", "explicit_certifiable", "mirror_partner")) <= set(row)
        assert row["name"]


def test_reason_mirror_pairs_and_uncertifiable_rows_are_exact():
    rows = load_reason_semantics(ROOT / "configs" / "precise_reason_semantics.yaml")
    partner = {row["id"]: row["mirror_partner"] for row in rows}
    assert partner[9] == 15 and partner[15] == 9
    assert partner[10] == 16 and partner[16] == 10
    assert partner[11] == 17 and partner[17] == 11
    assert partner[12] == 18 and partner[18] == 12
    assert partner[13] == 19 and partner[19] == 13
    assert partner[14] == 20 and partner[20] == 14
    for idx in (12, 13, 14, 18, 19, 20):
        assert rows[idx]["explicit_certifiable"] is False


def test_action_semantics_is_fixed_four_way_multilabel_contract():
    actions = load_action_semantics(ROOT / "configs" / "precise_action_semantics.yaml")
    assert [(row["id"], row["name"]) for row in actions] == [
        (0, "forward"), (1, "stop"), (2, "left"), (3, "right")
    ]
    assert actions[2]["query_base"] == actions[3]["query_base"] == "side_shared"
    assert actions[2]["side_embedding"] == "left"
    assert actions[3]["side_embedding"] == "right"


def test_schema_rejects_placeholder_reason_name(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("reasons:\n  - {id: 0, name: reason_0}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="21"):
        load_reason_semantics(path)
