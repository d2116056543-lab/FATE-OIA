from __future__ import annotations

import warnings
import subprocess
import sys

from PIL import Image
import torch

from fate_oia.transforms import AspectRatioLetterboxTransform, _to_tensor, audit_image_size


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


def test_pil_tensor_conversion_preserves_rgb_without_typed_storage_warning():
    image = Image.new("RGB", (2, 1))
    image.putdata([(255, 0, 0), (0, 128, 255)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tensor = _to_tensor(image)

    assert tensor.shape == (3, 1, 2)
    assert torch.allclose(tensor[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.allclose(
        tensor[:, 0, 1], torch.tensor([0.0, 128.0 / 255.0, 1.0])
    )
    # PyTorch emits this warning once per process, so run the warning contract
    # in a fresh interpreter instead of accidentally consuming it in a prior
    # image-transform test.
    fresh = subprocess.run(
        [
            sys.executable,
            "-W",
            "always",
            "-c",
            "from PIL import Image; from fate_oia.transforms import _to_tensor; "
            "_to_tensor(Image.new('RGB', (1, 1), 'white'))",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert fresh.returncode == 0
    assert "TypedStorage" not in fresh.stderr
