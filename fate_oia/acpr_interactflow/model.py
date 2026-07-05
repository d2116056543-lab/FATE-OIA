from __future__ import annotations

import math
import time
from dataclasses import replace

import torch
from torch import nn

from .decision_ledger import DecisionLedgerHead
from .dynamic_predicate_field import DynamicPredicateField
from .exp29_head import Exp29Head
from .interaction_flow import InteractionFlowReasoner
from .motion_path import MotionPathEncoder
from .nnpu_calalign import NNPUCalAlignHead
from .state_bank import ObjectiveEnvironmentStateBank
from .timing import StepTimer
from .types import ACPRInteractFlowPPOutput
from .visual_encoder import InteractVisualEncoder


class DirectVisualActionHead(nn.Module):
    """DINO-probe-style direct action path for PSI action classification."""

    def __init__(self, dim: int = 384, num_actions: int = 3, spatial_queries: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_actions = int(num_actions)
        self.spatial_queries_count = max(1, int(spatial_queries))
        self.spatial_queries = nn.Parameter(torch.randn(self.spatial_queries_count, self.dim) * 0.02)
        self.spatial_norm = nn.LayerNorm(self.dim)
        frame_dim = self.dim * (1 + self.spatial_queries_count)
        pooled_dim = frame_dim * 3
        self.token_proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.dim),
            nn.GELU(),
            nn.LayerNorm(self.dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.num_actions),
        )

    def forward(self, visual) -> dict[str, torch.Tensor]:
        cls_tokens = visual.cls_tokens
        patch_tokens = visual.patch_tokens_by_layer
        b, anchors, layers, patches, dim = patch_tokens.shape
        if dim != self.dim:
            raise ValueError(f"DirectVisualActionHead expected dim={self.dim}, got {dim}")
        dense_tokens = patch_tokens.reshape(b, anchors, layers * patches, dim)
        normalized = self.spatial_norm(dense_tokens)
        scores = torch.einsum("qd,band->baqn", self.spatial_queries, normalized) / math.sqrt(float(dim))
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("baqn,band->baqd", attention, dense_tokens).reshape(b, anchors, -1)
        frame_features = torch.cat([cls_tokens, attended], dim=-1)
        mean_feat = frame_features.mean(dim=1)
        last_feat = frame_features[:, -1]
        delta_feat = frame_features[:, -1] - frame_features[:, 0]
        decision_features = self.token_proj(torch.cat([mean_feat, last_feat, delta_feat], dim=-1))
        logits = self.action_head(decision_features)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
        return {"logits": logits, "features": decision_features, "attention_entropy": entropy}


def bounded_exp29_calalign_delta(calibrated_logits: torch.Tensor, raw_logits: torch.Tensor, max_delta: float = 0.05) -> torch.Tensor:
    """Keep learned Exp29 CalAlign from overriding train-only deploy theta."""
    return (calibrated_logits - raw_logits).clamp(-max_delta, max_delta)


class ACPRInteractFlowPPModel(nn.Module):
    """Formal PSI model. It does not instantiate legacy OIA model or dataset classes."""

    def __init__(
        self,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        predicate_config: str = "configs/acpr_interactflow_predicates.yaml",
        grammar_path: str = "configs/acpr_interactflow_state_grammar.yaml",
        exp29_names_path: str | None = None,
        oia_acpr_checkpoint: str | None = None,
        text_encoder_model: str | None = None,
        require_oia_transfer_source: bool = False,
        require_transformer_text: bool = False,
        dim: int = 384,
        action_dim: int = 3,
        use_mock_dino: bool = False,
        dino_chunk_size: int = 2,
        anchor_frames: tuple[int, ...] = (0, 3, 6, 9, 12, 14),
        selected_layers: tuple[int, ...] = (3, 7, 11),
        dino_input_height: int = 320,
        dino_input_width: int = 576,
        patch_size: int = 8,
    ) -> None:
        super().__init__()
        self.visual = InteractVisualEncoder(
            pretrained_weights=pretrained_weights,
            anchor_frames=anchor_frames,
            selected_layers=selected_layers,
            use_mock_dino=use_mock_dino,
            dim=dim,
            dino_chunk_size=dino_chunk_size,
            dino_input_height=dino_input_height,
            dino_input_width=dino_input_width,
            patch_size=patch_size,
        )
        self.motion = MotionPathEncoder(dim=dim)
        self.direct_action = DirectVisualActionHead(dim=self.visual.dim, num_actions=action_dim, spatial_queries=8)
        self.predicates = DynamicPredicateField(
            predicate_config=predicate_config,
            dim=dim,
            source_checkpoint=oia_acpr_checkpoint,
            text_encoder_model=text_encoder_model,
            require_source_checkpoint=require_oia_transfer_source,
            require_transformer_text=require_transformer_text,
        )
        self.state_bank = ObjectiveEnvironmentStateBank(grammar_path=grammar_path, dim=dim, num_predicates=48)
        self.flow = InteractionFlowReasoner(grammar_path=grammar_path, dim=dim, num_predicates=48, num_actions=action_dim)
        self.ledger = DecisionLedgerHead(dim=dim, num_actions=action_dim)
        self.exp29 = Exp29Head(dim=dim, exp_dim=29, label_names_path=exp29_names_path)
        self.calalign = NNPUCalAlignHead(action_dim=action_dim, exp_dim=29)

    def _factor_mask(self, name: str, device: torch.device) -> torch.Tensor:
        mask = torch.ones(self.flow.num_factors, device=device)
        factors = self.flow.grammar.flow_factors
        if name == "factor_off":
            mask.zero_()
        elif name == "regime_off":
            for i, item in enumerate(factors):
                text = str(item.get("name", ""))
                if any(key in text for key in ("clear", "caution", "yield", "stop")):
                    mask[i] = 0.0
        elif name == "phase_off":
            for i, item in enumerate(factors):
                text = str(item.get("name", ""))
                if any(key in text for key in ("waiting", "approaching", "entering", "crossing", "decelerating")):
                    mask[i] = 0.0
        elif name == "source_off":
            for i, item in enumerate(factors):
                source = str(item.get("source", ""))
                if source and source != "global_context":
                    mask[i] = 0.0
        return mask

    def forward(
        self,
        input_frames: torch.Tensor,
        epoch: int = 0,
        intervention: str | None = None,
        action_soft_target: torch.Tensor | None = None,
    ) -> ACPRInteractFlowPPOutput:
        timer = StepTimer()
        with timer.section("visual_dino"):
            visual = self.visual(input_frames)
        with timer.section("visual_motion"):
            motion = self.motion(visual.fast_motion_tokens)
            direct_action = self.direct_action(visual)
        with timer.section("predicate"):
            predicates = self.predicates(
                visual.patch_tokens_by_layer,
                lowres_motion_maps=visual.lowres_motion_maps,
                fast_motion_tokens=visual.fast_motion_tokens,
                grid_hw=visual.grid_hw,
            )
        if intervention in {"predicate_off", "evidence_tube_off"}:
            predicates = replace(
                predicates,
                predicate_logits=torch.zeros_like(predicates.predicate_logits),
                predicate_probs=torch.zeros_like(predicates.predicate_probs),
                predicate_logits_trajectory=torch.zeros_like(predicates.predicate_logits_trajectory),
                predicate_probs_trajectory=torch.zeros_like(predicates.predicate_probs_trajectory),
                predicate_tokens=torch.zeros_like(predicates.predicate_tokens),
                predicate_token_trajectory=torch.zeros_like(predicates.predicate_token_trajectory),
                predicate_attention=torch.zeros_like(predicates.predicate_attention),
                predicate_evidence_maps=torch.zeros_like(predicates.predicate_evidence_maps),
                predicate_confidence=torch.zeros_like(predicates.predicate_confidence),
                predicate_relative_motion=torch.zeros_like(predicates.predicate_relative_motion),
            )
        with timer.section("predicate"):
            state_bank = self.state_bank(predicates.predicate_tokens, predicates.predicate_probs)
        factor_mask = None
        if intervention in {"regime_off", "phase_off", "source_off", "factor_off"}:
            factor_mask = self._factor_mask(intervention, input_frames.device)
        flow_start = time.perf_counter()
        flow = self.flow(
            predicates.predicate_token_trajectory,
            predicates.predicate_probs_trajectory,
            motion["motion_token"],
            predicate_corridor_mass=predicates.predicate_corridor_mass,
            lag_disabled=(intervention == "lag_disabled"),
            factor_mask=factor_mask,
        )
        flow_total = time.perf_counter() - flow_start
        response_lag_time = float(flow.stats.get("response_lag_time", 0.0))
        timer.add("response_lag", response_lag_time)
        timer.add("interaction_flow", max(flow_total - response_lag_time, 0.0))
        visual_token = direct_action["features"]
        predicate_token = predicates.predicate_tokens.mean(1) + state_bank["state_tokens"].mean(1)
        with timer.section("decision_ledger"):
            ledger = self.ledger(
                visual_token,
                motion["motion_token"],
                predicate_token,
                flow.factor_tokens,
                flow.flow_edges,
                action_soft_target=action_soft_target,
                direct_action_logits=direct_action["logits"],
            )
        if intervention == "global_only":
            final_logits = ledger.global_logits + ledger.calibration_delta
            ledger = replace(
                ledger,
                gated_state_contributions=torch.zeros_like(ledger.gated_state_contributions),
                flow_delta_logits=torch.zeros_like(ledger.flow_delta_logits),
                final_logits=final_logits,
                identity_error=(final_logits - (ledger.global_logits + ledger.calibration_delta)).abs().max(),
            )
        with timer.section("exp29"):
            exp29 = self.exp29(
                factor_tokens_lag=flow.factor_tokens,
                predicate_tokens_summary=predicates.predicate_tokens,
                gated_state_contributions=ledger.gated_state_contributions,
                global_decision_hidden=ledger.global_hidden,
                action_logits=ledger.final_logits,
            )
            cal = self.calalign(ledger.final_logits, exp29.logits)
        exp29_logits_calibrated = exp29.logits_calibrated + bounded_exp29_calalign_delta(cal["exp29_logits_calibrated"], exp29.logits, max_delta=0.05)
        aux = {
            "action_logits_calibrated": cal["action_logits_calibrated"],
            "exp29_logits_calibrated": exp29_logits_calibrated,
            "state_group_logits": state_bank["state_group_logits"],
            "state_layer_weights": state_bank["state_layer_weights"],
            "state_attention": state_bank["state_attention"],
            "state_stats": state_bank["state_stats"],
            "motion_logits": motion["motion_logits"],
            "epoch": epoch,
            "action_logits_direct": direct_action["logits"],
            "action_direct_features": direct_action["features"],
            "action_direct_attention_entropy": direct_action["attention_entropy"],
        }
        aux["model_timing"] = timer.summary(reset=False)
        return ACPRInteractFlowPPOutput(
            action_logits=ledger.final_logits,
            action_probs=torch.softmax(ledger.final_logits, dim=-1),
            exp29_logits=exp29.logits,
            exp29_probs=torch.sigmoid(exp29_logits_calibrated),
            visual=visual,
            predicates=predicates,
            flow=flow,
            ledger=ledger,
            exp29=exp29,
            aux=aux,
        )
