from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ACPRReasonGrammar:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.actions: dict[int, dict[str, Any]] = {int(k): v for k, v in data.get("actions", {}).items()}
        self.reasons: dict[int, dict[str, Any]] = {int(k): v for k, v in data.get("reasons", {}).items()}
        if sorted(self.actions) != [0, 1, 2, 3]:
            raise ValueError("ACPR grammar must define exactly four actions 0..3")
        if sorted(self.reasons) != list(range(21)):
            raise ValueError("ACPR grammar must define exactly 21 reasons 0..20")
        for idx, row in self.reasons.items():
            name = str(row.get("name", ""))
            if name.startswith("reason_") or name in {"", str(idx)}:
                raise ValueError(f"Placeholder reason name at index {idx}")

    @property
    def reason_names(self) -> list[str]:
        return [str(self.reasons[i]["name"]) for i in range(21)]

    @property
    def action_names(self) -> list[str]:
        return [str(self.actions[i]["name"]) for i in range(4)]

    @property
    def tail_indices(self) -> list[int]:
        return [i for i in range(21) if bool(self.reasons[i].get("tail", False))]

    def predicate_names(self) -> list[str]:
        names: list[str] = []
        for i in range(21):
            names.extend(self.reasons[i].get("positive_predicates", []))
            names.extend(self.reasons[i].get("contradictory_predicates", []))
        return sorted(set(str(x) for x in names))

    def reason_predicate_matrix(self, predicate_names: list[str]) -> tuple[list[list[float]], list[list[float]]]:
        index = {name: i for i, name in enumerate(predicate_names)}
        pos = [[0.0 for _ in predicate_names] for _ in range(21)]
        neg = [[0.0 for _ in predicate_names] for _ in range(21)]
        for r in range(21):
            for name in self.reasons[r].get("positive_predicates", []):
                if name in index:
                    pos[r][index[name]] = 1.0
            for name in self.reasons[r].get("contradictory_predicates", []):
                if name in index:
                    neg[r][index[name]] = 1.0
        return pos, neg
