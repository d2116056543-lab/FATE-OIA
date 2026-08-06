from fate_oia.models.aie_oia_model import AIEOIAModel


def test_dino_is_frozen():
    model = AIEOIAModel(dim=32, mock_dim=32, use_mock_dino=True)
    assert all(not parameter.requires_grad for parameter in model.foundation.dino.parameters())

