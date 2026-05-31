from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def cache_key(file_name: str) -> str:
    return hashlib.sha1(str(file_name).encode("utf-8")).hexdigest()


class DinoTokenCache:
    def __init__(self, root: str | Path, image_height: int = 360, image_width: int = 640, arch: str = "vit_small", patch_size: int = 8, dtype: str = "fp16") -> None:
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.items = self.root / "items"; self.items.mkdir(parents=True, exist_ok=True)
        self.manifest = {"version": "trace_dino_cache_v1", "image_height": image_height, "image_width": image_width, "arch": arch, "patch_size": patch_size, "dtype": dtype}
        self._hits = 0; self._misses = 0

    def path_for(self, file_name: str) -> Path:
        return self.items / f"{cache_key(file_name)}.pt"

    def write_manifest(self, extra: dict[str, Any] | None = None) -> None:
        data = dict(self.manifest); data.update(extra or {})
        (self.root / "cache_manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def put(self, file_name: str, tokens: torch.Tensor, labels: torch.Tensor | None = None) -> None:
        torch.save({"file_name": file_name, "tokens": tokens.detach().cpu().half(), "labels": labels.detach().cpu() if labels is not None else None}, self.path_for(file_name))

    def get(self, file_name: str) -> dict[str, Any] | None:
        p = self.path_for(file_name)
        if not p.exists():
            self._misses += 1; return None
        self._hits += 1; return torch.load(p, map_location="cpu")

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {"cache_hits": self._hits, "cache_misses": self._misses, "cache_hit_rate": float(self._hits / total) if total else 0.0, "cache_root": str(self.root)}

    def audit(self, file_names: list[str]) -> dict[str, Any]:
        present = sum(1 for x in file_names if self.path_for(x).exists())
        return {"checked": len(file_names), "present": present, "cache_hit_rate": float(present / len(file_names)) if file_names else 0.0, "cache_root": str(self.root)}
