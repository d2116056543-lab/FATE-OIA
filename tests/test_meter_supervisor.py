import json
from pathlib import Path

from fate_oia.engine import supervise_acpr_meter_oia_foreground as supervisor
from fate_oia.engine.supervise_acpr_meter_oia_foreground import (
    FALLBACK_LADDER,
    PILOT_GATES,
)


def test_tesa_foreground_supervisor_forbids_hidden_launch():
    path = Path("scripts/FATE_OIA_acpr_meter_oia_v2_tesa_foreground.ps1")
    text = path.read_text(encoding="utf-8")
    assert "Start-Process" not in text
    assert "Start-Job" not in text
    assert "supervise_acpr_meter_oia_foreground" in text


def test_tesa_supervisor_requires_all_pilot_gates_and_uses_fixed_ladder():
    assert PILOT_GATES == tuple("ABCDEFGH")
    assert FALLBACK_LADDER == ((6, 5), (5, 6), (4, 8), (3, 10), (2, 15))


def test_tesa_supervisor_accepts_solution2_truth_table(tmp_path, monkeypatch):
    head = "clean-head"
    review = tmp_path / "review.json"
    pilot = tmp_path / "pilot.json"
    review.write_text(json.dumps({"git_head": head}), encoding="utf-8")
    pilot.write_text(
        json.dumps(
            {
                "git_head": head,
                "pass": True,
                "gates": {gate: False for gate in PILOT_GATES},
                "two_epoch_admission": {
                    "pass": True,
                    "mechanism_pass_count": 3,
                    "rules": {"min_mechanism_classes": 2},
                    "truth_table": {
                        "deterministic": {"pass": True},
                        "action": {"pass": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        supervisor,
        "_git",
        lambda *args: head if args == ("rev-parse", "HEAD") else "",
    )
    readiness = supervisor.validate_training_readiness(review, pilot)
    assert readiness["pilot"]["two_epoch_admission"]["pass"]
