from __future__ import annotations

from pathlib import Path
import torch
import yaml


class PMCalReasonFormulaBank:
    def __init__(self, grammar_path: str | Path, predicate_names: list[str]) -> None:
        self.grammar_path = str(grammar_path)
        self.predicate_names = list(predicate_names)
        data = yaml.safe_load(Path(grammar_path).read_text(encoding="utf-8")) or {}
        self.actions = data.get("actions", {})
        self.reasons = data.get("reasons", {})
        if len(self.actions) != 4:
            raise ValueError("PMCal requires exactly 4 actions")
        if len(self.reasons) != 21:
            raise ValueError("PMCal requires exactly 21 reasons")
        self.reason_names = [str(self.reasons[i].get("name", f"reason_{i}")) for i in range(21)]
        if any(name.startswith("reason_") for name in self.reason_names):
            raise ValueError("Placeholder reason names are forbidden")
        self.tail_indices = [12, 9, 5, 14, 6, 11, 10, 13]
        self._pred_index = {name: i for i, name in enumerate(self.predicate_names)}
        self._action_index = {str(self.actions[i]["name"]): i for i in range(4)}
        self.positive_matrix = self._build_predicate_matrix("positive_predicates")
        self.contradiction_matrix = self._build_predicate_matrix("contradictory_predicates")
        self.compatible_action_mat = self._build_action_matrix()
        self.hard_negative_matrix = self._build_hard_negative_matrix()

    def _build_predicate_matrix(self, field: str) -> torch.Tensor:
        mat = torch.zeros(21, len(self.predicate_names), dtype=torch.float32)
        for rid in range(21):
            for name in self.reasons[rid].get(field, []):
                idx = self._pred_index.get(str(name))
                if idx is not None:
                    mat[rid, idx] = 1.0
        return mat

    def _build_action_matrix(self) -> torch.Tensor:
        mat = torch.zeros(21, 4, dtype=torch.float32)
        for rid in range(21):
            for name in self.reasons[rid].get("compatible_actions", []):
                idx = self._action_index.get(str(name))
                if idx is not None:
                    mat[rid, idx] = 1.0
        return mat

    def _build_hard_negative_matrix(self) -> torch.Tensor:
        mat = torch.zeros(21, 21, dtype=torch.float32)
        for rid in range(21):
            for other in self.reasons[rid].get("hard_negative_reasons", []):
                j = int(other)
                if 0 <= j < 21:
                    mat[rid, j] = 1.0
        return mat
