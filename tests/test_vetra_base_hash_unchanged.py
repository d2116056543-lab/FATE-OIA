from fate_oia.engine.vetra_common import tensor_state_hash
from vetra_test_utils import build_model, fake_base


def test_base_parameter_hash_is_unchanged_after_decode_and_backward():
    model = build_model()
    before = tensor_state_hash(model.base_model)
    model.decode_base_output(fake_base(batch=2), alpha=1.0)["action_logits_final"].sum().backward()
    assert tensor_state_hash(model.base_model) == before

