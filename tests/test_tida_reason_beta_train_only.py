import inspect

from fate_oia.utils.tida_contracts import select_reason_beta


def test_reason_beta_selector_has_no_test_inputs():
    parameters = set(inspect.signature(select_reason_beta).parameters)
    assert not any("test" in name for name in parameters)
