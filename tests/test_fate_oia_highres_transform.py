from __future__ import annotations

from PIL import Image

from fate_oia.transforms import AspectRatioLetterboxTransform, audit_image_size


def test_letterbox_wide_image_keeps_16x9_without_padding():
    image = Image.new("RGB", (1280, 720), "white")
    transform = AspectRatioLetterboxTransform(360, 640, return_meta=True)
    tensor, meta = transform(image)
    assert tuple(tensor.shape[-2:]) == (360, 640)
    assert meta["original_size"] == (1280, 720)
    assert meta["resized_size"] == (640, 360)
    assert meta["padding"] == (0, 0, 0, 0)
    assert meta["patch_grid"] == (45, 80)


def test_letterbox_tall_image_records_padding_and_patch_grid():
    image = Image.new("RGB", (720, 1280), "white")
    transform = AspectRatioLetterboxTransform(360, 640, patch_size=8, return_meta=True)
    tensor, meta = transform(image)
    assert tuple(tensor.shape[-2:]) == (360, 640)
    assert meta["original_size"] == (720, 1280)
    assert meta["padding"][0] > 0
    assert meta["patch_grid"][0] * meta["patch_grid"][1] == (360 // 8) * (640 // 8)


def test_audit_image_size_uses_real_pil_dimensions():
    image = Image.new("RGB", (321, 123), "white")
    assert audit_image_size(image)["original_size"] == (321, 123)
