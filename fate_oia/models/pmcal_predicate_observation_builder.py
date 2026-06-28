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
        grammar_path: str,
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

    def _geometry_for_record(self, record: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = len(self.predicate_names)
        value = torch.zeros(m, device=device)
        mask = torch.zeros(m, device=device)
        reliability = torch.zeros(m, device=device)
        if not record:
            return value, mask, reliability
        text = str(record).lower()
        for i, pred in enumerate(self.predicates):
            sources = [str(x).lower() for x in pred.get("bdd100k_sources", [])]
            if any(src in text for src in sources if src not in {"global", "road"}):
                value[i] = 1.0
                mask[i] = 1.0
                reliability[i] = 0.15
        return value, mask, reliability.clamp(max=0.15)

    def build(
        self,
        *,
        file_names: list[str],
        reason_labels: torch.Tensor | None,
        structured_records: list[dict] | None,
        split: str,
        device: torch.device,
    ) -> dict[str, torch.Tensor | dict | list[str]]:
        b = len(file_names)
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
                "proxy_reliability_max": float(obs_geometry_reliability.max().detach().cpu()) if obs_geometry_reliability.numel() else 0.0,
            },
        }
