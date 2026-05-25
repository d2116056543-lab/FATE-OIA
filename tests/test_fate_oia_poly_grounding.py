from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch

from fate_oia.grounding.mask_builder import drivable_map_to_mask, objects_to_mask, poly2d_to_mask


def test_poly2d_to_mask_rasterizes_vertices():
    poly = {"vertices": [[0, 0], [8, 0], [8, 8], [0, 8]], "closed": True}
    mask = poly2d_to_mask([poly], image_size=(16, 16), output_size=(4, 4))
    assert mask.shape == (4, 4)
    assert float(mask.sum()) > 0


def test_objects_to_mask_combines_box_and_poly_categories():
    objects = [
        {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 8, "y2": 8}},
        {"category": "lane/crosswalk", "poly2d": [{"vertices": [[8, 8], [16, 8], [16, 16], [8, 16]], "closed": True}]},
    ]
    mask = objects_to_mask(objects, (16, 16), (4, 4), categories={"person", "lane/crosswalk"}, include_poly2d=True)
    assert mask.shape == (4, 4)
    assert float(mask.sum()) >= 4


def test_drivable_map_to_mask_loads_nonzero_pixels(tmp_path: Path):
    p = tmp_path / "drive.png"
    image = Image.new("L", (8, 8), 0)
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), 1)
    image.save(p)
    mask = drivable_map_to_mask(str(p), output_size=(4, 4))
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (4, 4)
    assert float(mask.sum()) > 0
