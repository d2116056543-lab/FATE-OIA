from test_tida_model_forward import _ImageBase

from fate_oia.models.tida_oia_model import TIDAOIAModel


def test_image_base_is_frozen_and_not_an_optimizer_owner():
    roles = {"static_anchor": [f"p{i}" for i in range(8)], "dynamic_actor": [f"p{i}" for i in range(8, 24)], "terminal_context": [f"p{i}" for i in range(24, 32)]}
    model = TIDAOIAModel(_ImageBase(), dim=8, predicate_roles=roles)
    assert all(not parameter.requires_grad for parameter in model.image_model.parameters())
    assert not any(name.startswith("image_model.") for name, parameter in model.named_parameters() if parameter.requires_grad)
