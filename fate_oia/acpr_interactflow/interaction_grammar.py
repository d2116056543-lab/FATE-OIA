from __future__ import annotations

from pathlib import Path

import yaml


class InteractionGrammar:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.state_groups = dict(data.get("state_groups", {}))
        self.response_lags = list(data.get("response_lags", [0, 1, 2, 3, 4]))
        self.flow_factors = list(data.get("flow_factors", []))
        if not self.flow_factors:
            raise ValueError("interaction grammar must define flow_factors")

