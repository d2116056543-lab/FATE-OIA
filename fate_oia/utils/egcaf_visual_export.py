from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw


def export_factor_overlay(image_path: str, factors: list[dict], output_path: str | Path) -> None:
    img = Image.open(image_path).convert("RGB") if Path(image_path).exists() else Image.new("RGB", (640, 360), "black")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for rec in factors[:12]:
        box = rec.get("box") or [0, 0, 1, 1]
        xy = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
        draw.rectangle(xy, outline="red", width=2)
        draw.text((xy[0], xy[1]), str(rec.get("type", rec.get("factor_index", "f"))), fill="yellow")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
