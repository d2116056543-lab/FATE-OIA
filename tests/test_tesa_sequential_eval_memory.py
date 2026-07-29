from fate_oia.engine.tesa_diagnostics import SEQUENTIAL_EVAL_MODES


def test_sequential_eval_order_is_fixed() -> None:
    assert SEQUENTIAL_EVAL_MODES == (
        "main", "visual", "factor_off", "reason_global", "reason_correction_off", "counterfactual"
    )
