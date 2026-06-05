from __future__ import annotations

import json
from pathlib import Path

import torch


class BDD100KSceneStateWeakLabelProvider:
    """Build weak scene-state labels from BDD100K object/lane/drivable geometry.

    Output order:
    traffic_control, front_object, vulnerable_user, left_lane, right_lane, drivable.
    Missing BDD100K files return available=False; this is still recorded.
    This provider builds file indexes once; it must not rglob per sample during training.
    """

    def __init__(self, bdd100k_root: str | Path | None) -> None:
        self.root = Path(bdd100k_root) if bdd100k_root else None
        self.cache: dict[str, tuple[torch.Tensor, bool]] = {}
        self.label_index: dict[str, Path] = {}
        self.drivable_stems: set[str] = set()
        if self.root is not None:
            label_root = self.root / "bdd100k_labels"
            if label_root.exists():
                self.label_index = {p.stem: p for p in label_root.rglob("*.json")}
            drive_root = self.root / "bdd100k_drivable_maps"
            if drive_root.exists():
                for p in drive_root.rglob("*.png"):
                    stem = p.stem.replace("_drivable_color", "").replace("_drivable_id", "")
                    self.drivable_stems.add(stem)

    @staticmethod
    def base_stem(file_name: str) -> str:
        stem = Path(file_name).stem
        for suffix in ("_1", "_3", "_5"):
            if stem.endswith(suffix):
                return stem[: -len(suffix)]
        return stem

    def _candidate_jsons(self, stem: str) -> list[Path]:
        p = self.label_index.get(stem)
        return [p] if p is not None else []

    @staticmethod
    def _box_center(obj: dict) -> tuple[float, float, float] | None:
        box = obj.get("box2d") or {}
        try:
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
        except Exception:
            return None
        cx = ((x1 + x2) / 2.0) / 1280.0
        cy = ((y1 + y2) / 2.0) / 720.0
        area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / (1280.0 * 720.0)
        return cx, cy, area

    @staticmethod
    def _poly_mean_x(obj: dict) -> float | None:
        polys = obj.get("poly2d") or []
        xs: list[float] = []
        for poly in polys:
            if isinstance(poly, dict):
                vertices = poly.get("vertices") or []
                for v in vertices:
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        xs.append(float(v[0]) / 1280.0)
            elif isinstance(poly, (list, tuple)) and len(poly) >= 2:
                # BDD100K common format: [x, y, control_type].
                try:
                    xs.append(float(poly[0]) / 1280.0)
                except Exception:
                    pass
        if not xs:
            return None
        return sum(xs) / len(xs)

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
                geom = self._box_center(obj)
                if geom is None or geom[2] > 1e-5:
                    vec[0] = 1
            if cat in {"car", "truck", "bus", "train", "motor", "bike"}:
                geom = self._box_center(obj)
                if geom is not None:
                    cx, cy, area = geom
                    if 0.32 <= cx <= 0.68 and cy >= 0.32 and area > 2e-4:
                        vec[1] = 1
            if cat in {"person", "rider"}:
                geom = self._box_center(obj)
                if geom is not None:
                    cx, cy, area = geom
                    if 0.20 <= cx <= 0.80 and cy >= 0.30 and area > 5e-5:
                        vec[2] = 1
            if "lane" in cat or "road curb" in cat:
                mx = self._poly_mean_x(obj)
                if mx is None:
                    vec[3] = max(vec[3], 0.5)
                    vec[4] = max(vec[4], 0.5)
                elif mx < 0.50:
                    vec[3] = 1
                else:
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
        if self.root is not None and stem in self.drivable_stems:
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
