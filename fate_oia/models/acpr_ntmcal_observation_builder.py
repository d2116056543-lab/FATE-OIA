from __future__ import annotations

from pathlib import Path

import torch
import yaml

from .acpr_ntmcal_predicate_bank import NativePredicateBank


class NativeTextObservationBuilder:
    def __init__(self, predicate_bank: NativePredicateBank, reason_formula_path: str | Path) -> None:
        self.bank = predicate_bank
        data = yaml.safe_load(Path(reason_formula_path).read_text(encoding="utf-8")) or {}
        self.reasons = sorted(data["reasons"], key=lambda r: int(r["id"]))
        if len(self.reasons) != 21 or [int(r["id"]) for r in self.reasons] != list(range(21)):
            raise ValueError("reason formulas must cover ids 0..20")
        self.tail_reason_indices = list(data.get("tail_reason_indices", []))
        self.support = torch.zeros(21, len(self.bank.specs))
        self.contra = torch.zeros(21, len(self.bank.specs))
        for r in self.reasons:
            rid = int(r["id"])
            for name in r.get("support_predicates", []):
                self.support[rid, self.bank.name_to_id[name]] = 1.0
            for name in r.get("contra_predicates", []):
                self.contra[rid, self.bank.name_to_id[name]] = 1.0

    def __call__(
        self,
        reason_labels: torch.Tensor | None,
        *,
        file_names=None,
        structured_records=None,
        split: str = "train",
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor | dict]:
        p = len(self.bank.specs)
        if split != "train" or reason_labels is None:
            b = int(batch_size or (reason_labels.shape[0] if reason_labels is not None else 1))
            dev = device or (reason_labels.device if reason_labels is not None else torch.device("cpu"))
            return {
                "obs_value": torch.zeros(b, p, device=dev),
                "obs_mask": torch.zeros(b, p, device=dev),
                "obs_soft_negative": torch.zeros(b, p, device=dev),
                "source_stats": {
                    "test_ignored": True,
                    "text_obs_positive_count": 0,
                    "text_obs_soft_negative_count": 0,
                    "text_obs_unknown_count": int(b * p),
                },
            }
        device = reason_labels.device
        support = self.support.to(device)
        contra = self.contra.to(device)
        pos = (reason_labels.float() @ support).clamp(0, 1)
        soft_neg = (reason_labels.float() @ contra).clamp(0, 1) * (1.0 - pos)
        mask = (pos + soft_neg).clamp(0, 1)
        value = pos
        unk = (1.0 - mask).clamp(0, 1)
        return {
            "obs_value": value,
            "obs_mask": mask,
            "obs_soft_negative": soft_neg,
            "source_stats": {
                "test_ignored": False,
                "text_obs_positive_count": int(pos.sum().detach().cpu()),
                "text_obs_soft_negative_count": int(soft_neg.sum().detach().cpu()),
                "text_obs_unknown_count": int(unk.sum().detach().cpu()),
            },
        }
