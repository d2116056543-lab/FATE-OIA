from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fate_oia.models.psr_calibration import apply_reason_temperature_bias
from fate_oia.utils.psr_metrics import compute_psr_metrics


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def evidence_reliability(selected_score: float | None, random_score: float | None) -> float:
    if selected_score is None or random_score is None:
        return 0.0
    return max(0.0, float(selected_score) - float(random_score))


@dataclass
class PSRRouterOutput:
    action_logits: torch.Tensor
    reason_logits: torch.Tensor
    alpha_action: torch.Tensor
    alpha_reason: torch.Tensor
    metadata: dict[str, Any]


class PSRFeatureBuilder:
    def build(
        self,
        action_a: torch.Tensor,
        reason_a: torch.Tensor,
        action_e: torch.Tensor,
        reason_e: torch.Tensor,
        reason_c: torch.Tensor | None = None,
        evidence_rel: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pa = torch.sigmoid(action_a)
        pe_a = torch.sigmoid(action_e)
        ra = torch.sigmoid(reason_a)
        re = torch.sigmoid(reason_e)
        rc = torch.sigmoid(reason_c) if reason_c is not None else ra
        action_features = torch.stack(
            [
                pa,
                pe_a,
                (pa - 0.5).abs(),
                (pe_a - 0.5).abs(),
                binary_entropy_from_logits(action_a),
                binary_entropy_from_logits(action_e),
                (pa - pe_a).abs(),
                torch.full_like(pa, float(evidence_rel)),
            ],
            dim=-1,
        )
        reason_features = torch.stack(
            [
                ra,
                re,
                rc,
                (ra - 0.5).abs(),
                (re - 0.5).abs(),
                binary_entropy_from_logits(reason_a),
                binary_entropy_from_logits(reason_e),
                (ra - re).abs(),
            ],
            dim=-1,
        )
        return action_features, reason_features


class StaticLabelRouter:
    def __init__(self, reason_source_by_label: list[str] | None = None, action_cap: float = 0.08):
        self.reason_source_by_label = reason_source_by_label
        self.action_cap = float(action_cap)

    def __call__(
        self,
        action_a: torch.Tensor,
        reason_a: torch.Tensor,
        action_e: torch.Tensor,
        reason_e: torch.Tensor,
        reason_c: torch.Tensor | None = None,
        evidence_rel: float = 0.0,
    ) -> PSRRouterOutput:
        alpha_action = torch.zeros_like(action_a)
        if evidence_rel > 0:
            low_conf = ((torch.sigmoid(action_a) - 0.5).abs() < 0.15).float()
            alpha_action = low_conf * 0.10
        candidate_action = action_a + alpha_action * (action_e - action_a).clamp(-self.action_cap, self.action_cap)
        reason_final = reason_a.clone()
        alpha_reason = torch.zeros_like(reason_a)
        source = self.reason_source_by_label or ["E"] * reason_a.shape[1]
        for i, s in enumerate(source):
            if s == "E":
                reason_final[:, i] = reason_e[:, i]
                alpha_reason[:, i] = 1.0
            elif s == "C" and reason_c is not None:
                reason_final[:, i] = reason_c[:, i]
                alpha_reason[:, i] = 0.5
        return PSRRouterOutput(candidate_action, reason_final, alpha_action, alpha_reason, {"router": "static_label", "reason_source_by_label": source})


class DynamicMarginEntropyRouter:
    def __init__(self, reason_margin_delta: float = 0.05, evidence_action_entropy_threshold: float = 0.62, action_cap: float = 0.08):
        self.reason_margin_delta = float(reason_margin_delta)
        self.evidence_action_entropy_threshold = float(evidence_action_entropy_threshold)
        self.action_cap = float(action_cap)

    def __call__(
        self,
        action_a: torch.Tensor,
        reason_a: torch.Tensor,
        action_e: torch.Tensor,
        reason_e: torch.Tensor,
        evidence_rel: float = 0.0,
    ) -> PSRRouterOutput:
        margin_a = (torch.sigmoid(reason_a) - 0.5).abs()
        margin_e = (torch.sigmoid(reason_e) - 0.5).abs()
        alpha_reason = (margin_e > (margin_a + self.reason_margin_delta)).float()
        action_entropy = binary_entropy_from_logits(action_a)
        alpha_action = torch.zeros_like(action_a)
        if evidence_rel > 0:
            alpha_action = (action_entropy > self.evidence_action_entropy_threshold).float() * 0.10
        action_final = action_a + alpha_action * (action_e - action_a).clamp(-self.action_cap, self.action_cap)
        reason_final = alpha_reason * reason_e + (1.0 - alpha_reason) * reason_a
        return PSRRouterOutput(action_final, reason_final, alpha_action, alpha_reason, {"router": "dynamic_margin_entropy"})


class LearnedPSRRouter(nn.Module):
    def __init__(self, hidden_dim: int = 64, dropout: float = 0.05):
        super().__init__()
        self.action_mlp = nn.Sequential(nn.Linear(8, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.reason_mlp = nn.Sequential(nn.Linear(8, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.reason_bias = nn.Parameter(torch.zeros(21))
        self.reason_log_temp = nn.Parameter(torch.zeros(21))

    def forward(
        self,
        action_features: torch.Tensor,
        reason_features: torch.Tensor,
        action_a: torch.Tensor,
        reason_a: torch.Tensor,
        action_e: torch.Tensor,
        reason_e: torch.Tensor,
        action_cap: float = 0.08,
    ) -> PSRRouterOutput:
        alpha_action = torch.sigmoid(self.action_mlp(action_features).squeeze(-1))
        alpha_reason = torch.sigmoid(self.reason_mlp(reason_features).squeeze(-1))
        action_final = action_a + alpha_action * (action_e - action_a).clamp(-action_cap, action_cap)
        reason_mix = alpha_reason * reason_e + (1.0 - alpha_reason) * reason_a
        reason_final = apply_reason_temperature_bias(reason_mix, self.reason_log_temp.exp(), self.reason_bias)
        return PSRRouterOutput(action_final, reason_final, alpha_action, alpha_reason, {"router": "learned"})


class ParetoSafetySelector:
    def __init__(self, action_epsilon: float = 0.0, reason_map_epsilon: float = 0.002):
        self.action_epsilon = float(action_epsilon)
        self.reason_map_epsilon = float(reason_map_epsilon)

    def guard_action(
        self,
        candidate_action: torch.Tensor,
        base_action: torch.Tensor,
        reason_logits: torch.Tensor,
        labels_action: torch.Tensor,
        labels_reason: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        cand = compute_psr_metrics(candidate_action, reason_logits, labels_action, labels_reason).to_dict()
        base = compute_psr_metrics(base_action, reason_logits, labels_action, labels_reason).to_dict()
        if cand["Act_mF1"] + self.action_epsilon < base["Act_mF1"]:
            return base_action, {"pareto_action_fallback": True, "candidate_Act_mF1": cand["Act_mF1"], "base_Act_mF1": base["Act_mF1"]}
        return candidate_action, {"pareto_action_fallback": False, "candidate_Act_mF1": cand["Act_mF1"], "base_Act_mF1": base["Act_mF1"]}
