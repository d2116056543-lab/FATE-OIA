from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.tida_context_encoder import TIDAContextEncoder


def test_context_state_dict_does_not_register_dino_twice():
    dino = ACPRDinoFieldExtractor(use_mock_dino=True, mock_dim=8)
    context = TIDAContextEncoder(dino_extractor=dino, query_reader=None)
    assert not any("backbone" in key or "dino" in key for key in context.state_dict())
