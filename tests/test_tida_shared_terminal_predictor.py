from fate_oia.models.tida_terminal_innovation import TIDATerminalInnovation


def test_one_predictor_object_serves_both_branches():
    module = TIDATerminalInnovation(dim=16)
    assert module.history_predictor is module.no_history_predictor
