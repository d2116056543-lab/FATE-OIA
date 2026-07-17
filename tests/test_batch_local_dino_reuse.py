import torch

from fate_oia.models.mosaic_batch_field_reuse import BatchLocalDinoFieldReuse


def test_batch_local_dino_field_is_computed_once():
    calls = {"n": 0}

    def extractor(images):
        calls["n"] += 1
        return {"patch_tokens_by_layer": images}

    reuse = BatchLocalDinoFieldReuse(extractor)
    images = torch.randn(2, 3, 4, 4)
    first = reuse(images)
    second = reuse(images)
    assert first is second
    assert calls["n"] == 1

