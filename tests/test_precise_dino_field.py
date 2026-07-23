import torch
import vision_transformer as vits

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


def test_real_dino_loader_requires_exact_official_state_and_records_identity(tmp_path):
    reference = vits.vit_small(patch_size=8, num_classes=0)
    path = tmp_path / "official.pth"
    torch.save(reference.state_dict(), path)
    model = PRECISEDinoFieldExtractor(pretrained_weights=str(path))
    assert len(model.loaded_state_keys) == len(reference.state_dict())
    assert model.missing_keys == ()
    assert model.unexpected_keys == ()
    assert len(model.pretrained_weights_sha256) == 64

    broken = dict(reference.state_dict())
    broken.pop(next(iter(broken)))
    broken_path = tmp_path / "broken.pth"
    torch.save(broken, broken_path)
    try:
        PRECISEDinoFieldExtractor(pretrained_weights=str(broken_path))
    except RuntimeError as error:
        assert "exactly match" in str(error)
    else:
        raise AssertionError("incomplete DINO weights were silently accepted")
