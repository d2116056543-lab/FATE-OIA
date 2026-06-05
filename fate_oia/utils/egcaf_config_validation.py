from __future__ import annotations

from typing import Any


def validate_egcaf_config(cfg: dict[str, Any]) -> None:
    if cfg.get("model") not in {"egcaf_oia_v1", None}:
        raise ValueError("EG-CAF config must use model=egcaf_oia_v1")
    if cfg.get("no_feature_cache") is not True:
        raise ValueError("EG-CAF requires no_feature_cache=true")
    if cfg.get("test_only_eval") is not True:
        raise ValueError("EG-CAF requires test_only_eval=true")
    size = cfg.get("image_size") or [cfg.get("image_height"), cfg.get("image_width")]
    if list(size) != [360, 640]:
        raise ValueError(f"EG-CAF formal config requires image_size [360,640], got {size}")
    if cfg.get("best_selection_split", "test") != "test":
        raise ValueError("EG-CAF best selection must use test for this user-requested run")
