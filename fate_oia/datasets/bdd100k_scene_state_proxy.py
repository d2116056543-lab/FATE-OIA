from __future__ import annotations

import json
from pathlib import Path

import torch


class BDD100KSceneStateWeakLabelProvider:
    """Build weak scene-state labels from BDD100K object/lane/drivable geometry.

    Output order:
    traffic_control, front_object, vulnerable_user, left_lane, right_lane, drivable.
    Missing BDD100K files return available=False; this is still recorded.
    """

    def __init__(self, bdd100k_root: str | Path | None) -> None:
        self.root = Path(bdd100k_root) if bdd100k_root else None
        self.cache: dict[str, tuple[torch.Tensor, bool]] = {}

    @staticmethod
    def base_stem(file_name: str) -> str:
        stem = Path(file_name).stem
        for suffix in ("_1", "_3", "_5"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _candidate_jsons(self, stem: str) -> list[Path]:
        if self.root is None:
            return []
        label_root = self.root / "bdd100k_labels"
        return list(label_root.rglob(f"{stem}.json"))[:4]

    def _from_json(self, path: Path) -> torch.Tensor:
        vec = torch.zeros(6, dtype=torch.float32)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return vec
        frames = data.get("frames") or []
        objects = []
        for frame in frames:
            objects.extend(frame.get("objects") or [])
        for obj in objects:
            cat = str(obj.get("category", "")).lower()
            if "traffic light" in cat or "traffic sign" in cat or "stop" in cat:
                vec[0] = 1
            if cat in {"car", "truck", "bus", "train", "motor", "bike"}:
                vec[1] = 1
            if cat in {"person", "rider"}:
                vec[2] = 1
            if "lane" in cat or "road curb" in cat:
                vec[3] = 1
                vec[4] = 1
            if "drivable" in cat or "area/drivable" in cat:
                vec[5] = 1
        return vec

    def lookup(self, file_name: str) -> tuple[torch.Tensor, bool]:
        stem = self.base_stem(file_name)
        if stem in self.cache:
            return self.cache[stem]
        vec = torch.zeros(6, dtype=torch.float32)
        available = False
        for path in self._candidate_jsons(stem):
            vec = torch.maximum(vec, self._from_json(path))
            available = True
        # Drivable map presence is also a weak drivable signal.
        if self.root is not None:
            drive_root = self.root / "bdd100k_drivable_maps"
            if any(drive_root.rglob(f"{stem}*.png")):
                vec[5] = 1
                available = True
        self.cache[stem] = (vec, available)
        return self.cache[stem]

    def batch(self, file_names: list[str], device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        avail = []
        for name in file_names:
            vec, ok = self.lookup(name)
            rows.append(vec)
            avail.append(ok)
        return torch.stack(rows).to(device), torch.tensor(avail, dtype=torch.bool, device=device)



def build_scene_state_proxy(bdd100k_root=None):
    """Backward-compatible factory for tests and older launchers."""
    return BDD100KSceneStateWeakLabelProvider(bdd100k_root)
