from __future__ import annotations

from pathlib import Path


def build_atlas(case_dirs: list[str | Path], output_html: str | Path) -> None:
    links = "\n".join(f"<li>{Path(p).name}</li>" for p in case_dirs)
    Path(output_html).write_text(f"<html><body><ul>{links}</ul></body></html>", encoding="utf-8")

