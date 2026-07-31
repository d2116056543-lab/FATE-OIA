from fate_oia.optim.heca_optimization import validate_formal_protocol


def test_formal_run_rejects_weights_only_or_pilot_resume() -> None:
    validate_formal_protocol({"from_scratch": True, "epochs": 14, "pilot_checkpoint": None})
    for bad in (
        {"from_scratch": False, "epochs": 14, "pilot_checkpoint": None},
        {"from_scratch": True, "epochs": 14, "pilot_checkpoint": "pilot.pth"},
    ):
        try:
            validate_formal_protocol(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid formal protocol was accepted")

