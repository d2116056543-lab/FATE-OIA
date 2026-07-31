import json

import pytest

from fate_oia.engine import supervise_meter_oia_v3_heca_foreground as supervisor


def test_full_supervisor_requires_clean_matching_a_to_g(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(supervisor, "_git", lambda *args: "head" if args[0] == "rev-parse" else "")
    review = tmp_path / "review.json"
    pilot = tmp_path / "pilot.json"
    gate_c = tmp_path / "gate_c.json"
    review.write_text(json.dumps({"pass": True, "git_head": "head"}), encoding="utf-8")
    pilot.write_text(json.dumps({"pass": True, "git_head": "head", "gates": {letter: True for letter in "ABCDEFG"}}), encoding="utf-8")
    gate_c.write_text(json.dumps({"gate": "C", "pass": True}), encoding="utf-8")
    assert supervisor.validate_training_readiness(review, pilot, gate_c)["git_head"] == "head"
    payload = json.loads(pilot.read_text(encoding="utf-8"))
    payload["gates"]["E"] = False
    pilot.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        supervisor.validate_training_readiness(review, pilot, gate_c)
