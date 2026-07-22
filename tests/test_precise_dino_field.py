import torch

from fate_oia.models.precise_dino_field import PRECISEDinoFieldExtractor


def test_mock_dino_field_is_frozen_full_resolution_and_single_call():
    model = PRECISEDinoFieldExtractor(use_mock_dino=True)
    image = torch.randn(1, 3, 360, 640)
    output = model(image)
    assert output["patch_tokens_by_layer"].shape == (1, 3, 3600, 384)
    assert output["cls_tokens_by_layer"].shape == (1, 3, 384)
    assert output["grid_hw"] == (45, 80)
    assert output["original_tokens"] == 3601
    assert output["dino_call_count"].item() == 1
    assert all(not parameter.requires_grad for parameter in model.dino.parameters())


def test_dino_rejects_non_contract_image_shape():
    model = PRECISEDinoFieldExtractor(use_mock_dino=True)
    try:
        model(torch.randn(1, 3, 224, 224))
    except ValueError as error:
        assert "360x640" in str(error)
    else:
        raise AssertionError("non-contract image shape was accepted")


def test_frozen_dino_remains_eval_when_parent_enters_train_mode():
    model = PRECISEDinoFieldExtractor(use_mock_dino=True)
    model.train()
    assert model.training is False
    assert model.dino.training is False
