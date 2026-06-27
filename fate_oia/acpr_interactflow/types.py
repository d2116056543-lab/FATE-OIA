from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class PSIInteractFlowBatch:
    input_frames: torch.Tensor
    action_soft: torch.Tensor
    action_majority: torch.Tensor
    exp29: torch.Tensor
    exp29_mask: torch.Tensor
    paper_effective_weight: torch.Tensor
    video_id: list[str]
    start_frame: torch.Tensor
    target_frame_index: torch.Tensor
    target_frame_path: list[str]
    frame_paths: list[list[str]]
    explanation_text: list[str]
    reasoning_text: list[str]
    sample_id: list[str]
    meta: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InteractVisualOutput:
    anchor_tokens: torch.Tensor
    fast_motion_tokens: torch.Tensor
    lowres_motion_maps: torch.Tensor
    motion_tokens: torch.Tensor
    cls_tokens: torch.Tensor
    patch_tokens_by_layer: torch.Tensor
    grid_hw: tuple[int, int]
    anchor_indices: list[int]
    stats: dict[str, Any]


@dataclass
class InteractPredicateField:
    predicate_logits: torch.Tensor
    predicate_probs: torch.Tensor
    predicate_logits_trajectory: torch.Tensor
    predicate_probs_trajectory: torch.Tensor
    predicate_tokens: torch.Tensor
    predicate_token_trajectory: torch.Tensor
    predicate_attention: torch.Tensor
    predicate_evidence_maps: torch.Tensor
    predicate_confidence: torch.Tensor
    predicate_centroids: torch.Tensor
    predicate_relative_motion: torch.Tensor
    predicate_corridor_mass: torch.Tensor
    transfer_gate: torch.Tensor
    predicate_names: list[str]
    temporal_stats: dict[str, Any]


@dataclass
class InteractionFlowState:
    state_tokens: torch.Tensor
    state_logits: torch.Tensor
    state_attention: torch.Tensor
    factor_tokens_trajectory: torch.Tensor
    factor_logits_trajectory: torch.Tensor
    factor_probs_trajectory: torch.Tensor
    factor_to_predicate: torch.Tensor
    factor_to_corridor: torch.Tensor
    lag_weights: torch.Tensor
    flow_edges: torch.Tensor
    factor_tokens: torch.Tensor
    stats: dict[str, Any]


@dataclass
class InteractionDecisionLedger:
    global_logits: torch.Tensor
    visual_logits: torch.Tensor
    motion_logits: torch.Tensor
    predicate_logits: torch.Tensor
    raw_state_contributions: torch.Tensor
    gated_state_contributions: torch.Tensor
    flow_delta_logits: torch.Tensor
    calibration_delta: torch.Tensor
    final_logits: torch.Tensor
    gate: torch.Tensor
    benefit_gate: torch.Tensor
    benefit_target: torch.Tensor | None
    contribution_attention: torch.Tensor
    global_hidden: torch.Tensor
    contribution_terms: dict[str, torch.Tensor]
    identity_error: torch.Tensor


@dataclass
class Exp29Output:
    logits_raw: torch.Tensor
    logits_calibrated: torch.Tensor
    probs_raw: torch.Tensor
    probs_calibrated: torch.Tensor
    label_mask: torch.Tensor
    label_names: list[str]
    cluster_attention_to_factors: torch.Tensor
    cluster_reliability: torch.Tensor
    cluster_to_state_prior: torch.Tensor
    attention: torch.Tensor
    stats: dict[str, Any]

    @property
    def logits(self) -> torch.Tensor:
        return self.logits_raw

    @property
    def probs(self) -> torch.Tensor:
        return self.probs_raw


@dataclass
class ACPRInteractFlowPPOutput:
    action_logits: torch.Tensor
    action_probs: torch.Tensor
    exp29_logits: torch.Tensor
    exp29_probs: torch.Tensor
    visual: InteractVisualOutput
    predicates: InteractPredicateField
    flow: InteractionFlowState
    ledger: InteractionDecisionLedger
    exp29: Exp29Output
    aux: dict[str, Any]
