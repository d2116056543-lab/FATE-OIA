from __future__ import annotations

from pathlib import Path

import json


def load_exp29_names(path: str | Path | None = None) -> list[str]:
    if path:
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8")) if p.suffix.lower() == ".json" else None
            if isinstance(data, list) and len(data) == 29:
                return [str(x) for x in data]
            if isinstance(data, dict):
                names = data.get("label_names") or data.get("names")
                if isinstance(names, list) and len(names) == 29:
                    return [str(x) for x in names]
    return [f"psi_exp_cluster_{i:02d}" for i in range(29)]

