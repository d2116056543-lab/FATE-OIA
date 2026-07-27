"""Behavioral contract for the host's explicit BDD100K per-frame layout."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import types


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if HERE.parent.name == "tests" else HERE.parents[2]
P1_ROOT = ROOT / ".codex" / "p1_pickle_current"
STAGING = ROOT / "remote_patch" / "P21"
INDEX = STAGING / "bdd100k_task_aware_index.py" if STAGING.is_dir() else ROOT / "fate_oia" / "datasets" / "bdd100k_task_aware_index.py"


def _index_module():
    # The staging checkout deliberately contains only the P21 overlay.  Supply
    # the tiny, deterministic BDD-OIA stem contract so this test executes the
    # real per-frame parser rather than downgrading to a source-string check.
    if STAGING.is_dir():
        fate = sys.modules.setdefault("fate_oia", types.ModuleType("fate_oia"))
        fate.__path__ = [str(P1_ROOT / "fate_oia")]
        datasets = sys.modules.setdefault("fate_oia.datasets", types.ModuleType("fate_oia.datasets"))
        datasets.__path__ = [str(P1_ROOT / "fate_oia" / "datasets")]
        grounding = types.ModuleType("fate_oia.datasets.bdd100k_grounding")
        grounding.bdd_oia_base_stem = lambda name: Path(str(name)).stem.rsplit("_", 1)[0] if Path(str(name)).stem.rsplit("_", 1)[-1] in {"1", "3"} else Path(str(name)).stem
        sys.modules[grounding.__name__] = grounding
    spec = importlib.util.spec_from_file_location("rael_p21_grounding_index", INDEX)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_frame(path: Path, name: str) -> None:
    payload = {
        "name": name,
        "frames": [{"objects": [
            {"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 20, "y2": 30}},
            {"category": "lane/solid", "poly2d": [[1, 1, "L"], [2, 2, "L"]]},
            {"category": "area/drivable", "poly2d": [[0, 0, "L"], [3, 0, "L"], [3, 3, "L"]]},
        ]}]}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_explicit_per_frame_label_directories_split_object_types_and_preserve_suffix_lookup(tmp_path: Path) -> None:
    module = _index_module()
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    _write_frame(train / "frame.json", "clip_1.jpg")
    _write_frame(val / "frame.json", "valclip_1.jpg")
    index = module.RAELTaskAwareBDD100KIndex(label_directories={"train": train, "val": val})
    record = index.lookup("clip_3.jpg")
    assert len(record.detections) == 1
    assert len(record.lanes) == 1
    assert len(record.drivable) == 1
    assert record.drivable[0]["polygon"] == ((0.0, 0.0), (3.0, 0.0), (3.0, 3.0))
    assert record.lanes[0]["points"] == ((1.0, 1.0), (2.0, 2.0))
    assert all(record.source_complete.values())


def test_required_stem_filter_skips_unrelated_per_frame_json(tmp_path: Path) -> None:
    module = _index_module()
    train, val = tmp_path / "train", tmp_path / "val"
    train.mkdir()
    val.mkdir()
    _write_frame(train / "keep.json", "keep.jpg")
    _write_frame(train / "skip.json", "skip.jpg")
    _write_frame(val / "val.json", "val.jpg")
    index = module.RAELTaskAwareBDD100KIndex(
        label_directories={"train": train, "val": val},
        include_file_names=("keep_1.jpg",),
    )
    assert index.lookup("keep_1.jpg").source_complete["detections"] is True
    assert index.lookup("skip_1.jpg").source_complete["detections"] is False
    assert index.manifest()["filtered_file_count"] == 2
