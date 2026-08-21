import inspect

from fate_oia.models.tida_action_reader import TIDAActionReader


def test_action_reader_has_no_reason_or_text_value_input():
    parameters = inspect.signature(TIDAActionReader.forward).parameters
    forbidden = {"reason_logits", "reason_labels", "reason_tokens", "text_embeddings"}
    assert not forbidden.intersection(parameters)


def test_role_key_state_does_not_change_factor_values():
    import torch

    model = TIDAActionReader(dim=8, num_actions=4, num_predicates=32)
    action = torch.randn(2, 4, 8)
    predicate = torch.randn(2, 32, 8)
    innovation = torch.randn(2, 4, 8)
    reliability = torch.ones(2, 36)
    first = model(action, predicate, innovation, reliability, temporal_scale=1.0, predicate_key_state=predicate)
    second = model(action, predicate, innovation, reliability, temporal_scale=1.0, predicate_key_state=predicate + 10.0)
    torch.testing.assert_close(first["action_factor_value"], second["action_factor_value"])
