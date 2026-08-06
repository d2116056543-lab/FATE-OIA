import torch

from fate_oia.models.aie_oia_model import AIEOIAModel


def test_model_calls_dino_once_for_full_forward():
    model = AIEOIAModel(dim=32, use_mock_dino=True, mock_dim=32)
    calls = 0
    original = model.foundation.dino.forward
    def counted(images):
        nonlocal calls
        calls += 1
        return original(images)
    model.foundation.dino.forward = counted
    model(torch.randn(1, 3, 360, 640))
    assert calls == 1

