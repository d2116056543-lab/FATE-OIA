from __future__ import annotations

from pathlib import Path
from typing import Any
import torch
import yaml

from .pmcal_reason_formula_bank import PMCalReasonFormulaBank


class PMCalPredicateObservationBuilder:
    def __init__(
        self,
        scene_config: str,
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        bdd100k_root: str | None = None,
        text_prompt_config: str | None = None,
    ) -> None:
        self.scene_config = str(scene_config)
        self.grammar_path = str(grammar_path)
        self.bdd100k_root = bdd100k_root
        data = yaml.safe_load(Path(scene_config).read_text(encoding="utf-8")) or {}
        self.predicates = list(data.get("predicates", []))
        self.predicate_names = [str(p["name"]) for p in self.predicates]
        self.bank = PMCalReasonFormulaBank(grammar_path, self.predicate_names)

    @staticmethod
    def _flatten_categories(record: dict[str, Any]) -> set[str]:
        cats: set[str] = set()

        def visit(obj: Any) -> None:
            if isinstance(obj, dict):
                for key in ("category", "name", "label", "type"):
                    value = obj.get(key)
                    if isinstance(value, str):
                        cats.add(value.lower())
                for value in obj.values():
                    visit(value)
            elif isinstance(obj, list):
                for item in obj:
                    visit(item)

        visit(record)
        return cats

    @staticmethod
    def _has_key_recursive(record: Any, names: set[str]) -> bool:
        if isinstance(record, dict):
            for key, value in record.items():
                if key.lower() in names:
                    return True
                if PMCalPredicateObservationBuilder._has_key_recursive(value, names):
                    return True
        if isinstance(record, list):
            return any(PMCalPredicateObservationBuilder._has_key_recursive(x, names) for x in record)
        return False

    @staticmethod
    def _record_has_polyline(record: Any) -> bool:
        return PMCalPredicateObservationBuilder._has_key_recursive(record, {"poly2d", "vertices", "lane", "lanes", "lane_labels"})

    @staticmethod
    def _record_has_drivable(record: Any) -> bool:
        text = str(record).lower()
        return "drivable" in text or "direct" in text or "alternative" in text

    @staticmethod
    def _record_has_box(record: Any) -> bool:
        return PMCalPredicateObservationBuilder._has_key_recursive(record, {"box2d", "bbox", "box"})

    def _source_supported(self, sources: list[str], categories: set[str], record: dict[str, Any]) -> bool:
        text = str(record).lower()
        for src in sources:
            src = src.lower()
            if src in {"global", "road", "weather", "sky"}:
                continue
            if src == "lane" and self._record_has_polyline(record):
                return True
            if src == "drivable" and self._record_has_drivable(record):
                return True
            if src in categories:
                return True
            if src in text and (self._record_has_box(record) or src in {"crosswalk", "intersection"}):
                return True
        return False

    def _geometry_for_record(self, record: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = len(self.predicate_names)
        value = torch.zeros(m, device=device)
        mask = torch.zeros(m, device=device)
        reliability = torch.zeros(m, device=device)
        if not record:
            return value, mask, reliability
        categories = self._flatten_categories(record)
        for i, pred in enumerate(self.predicates):
            sources = [str(x).lower() for x in pred.get("bdd100k_sources", [])]
            if self._source_supported(sources, categories, record):
                value[i] = 1.0
                mask[i] = 1.0
                reliability[i] = 0.15
        return value, mask, reliability.clamp(max=0.15)

    def build(
        self,
        *,
        file_names: list[str] | None = None,
        batch_size: int | None = None,
        reason_labels: torch.Tensor | None = None,
        structured_records: list[dict] | None = None,
        split: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor | dict | list[str]]:
        b = int(batch_size if batch_size is not None else len(file_names or []))
        file_names = file_names or ["" for _ in range(b)]
        m = len(self.predicate_names)
        obs_reason_value = torch.zeros(b, m, device=device)
        obs_reason_mask = torch.zeros(b, m, device=device)
        obs_geometry_value = torch.zeros(b, m, device=device)
        obs_geometry_mask = torch.zeros(b, m, device=device)
        obs_geometry_reliability = torch.zeros(b, m, device=device)
        if split != "test" and reason_labels is not None:
            pos = self.bank.positive_matrix.to(device)
            obs_reason_value = (reason_labels.float() @ pos).clamp(0, 1)
            obs_reason_mask = (reason_labels.float() @ pos > 0).float()
        if split != "test" and structured_records is not None:
            vals, masks, rels = [], [], []
            for rec in structured_records:
                v, ma, re = self._geometry_for_record(rec or {}, device)
                vals.append(v)
                masks.append(ma)
                rels.append(re)
            if vals:
                obs_geometry_value = torch.stack(vals, 0)
                obs_geometry_mask = torch.stack(masks, 0)
                obs_geometry_reliability = torch.stack(rels, 0).clamp(max=0.15)
        geometry_positive_count = int((obs_geometry_value * obs_geometry_mask).sum().detach().cpu())
        return {
            "obs_reason_value": obs_reason_value,
            "obs_reason_mask": obs_reason_mask,
            "obs_geometry_value": obs_geometry_value,
            "obs_geometry_mask": obs_geometry_mask,
            "obs_geometry_reliability": obs_geometry_reliability,
            "obs_text_positive_names": self.predicate_names,
            "obs_text_negative_names": [f"no_{n}" for n in self.predicate_names],
            "source_stats": {
                "split": split,
                "reason_mask_sum": float(obs_reason_mask.sum().detach().cpu()),
                "geometry_mask_sum": float(obs_geometry_mask.sum().detach().cpu()),
                "geometry_positive_count": geometry_positive_count,
                "proxy_reliability_max": float(obs_geometry_reliability.max().detach().cpu()) if obs_geometry_reliability.numel() else 0.0,
            },
        }
