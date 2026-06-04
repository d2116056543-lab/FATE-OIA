from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.p3le_action_set_head import ActionSetHead
from fate_oia.models.p3le_evidence_bag import WeakEvidenceBagRegularizer
from fate_oia.models.p3le_pair_head import PairAwareTensorHead
from fate_oia.models.p3le_progressive_experts import ProgressiveLayeredExperts
from fate_oia.models.p3le_reason_reliability import ReasonReliabilityHead
from fate_oia.models.p3le_router import ParetoSafeRouter
from fate_oia.models.p3le_shared_encoder import P3LESharedLabelQueryEncoder


class P3LEPairOIAFeatureModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        tail_indices: tuple[int, ...] = (5, 6, 9, 10, 11, 12, 13, 14),
        action_residual_cap: float = 0.04,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.shared_encoder = P3LESharedLabelQueryEncoder(dim, action_dim, reason_dim)
        self.ple = ProgressiveLayeredExperts(dim, action_dim, reason_dim, tail_indices=tail_indices)
        self.action_head = nn.Linear(dim, 1)
        self.reason_head = nn.Linear(dim, 1)
        self.action_aux_reason = nn.Linear(dim, 1)
        self.reason_aux_action = nn.Linear(dim, 1)
        self.pair_head = PairAwareTensorHead(dim, action_dim, reason_dim, rank=32)
        self.action_set_head = ActionSetHead(dim, action_dim, num_prototypes=8)
        self.evidence_bag = WeakEvidenceBagRegularizer(dim, reason_dim)
        self.reliability = ReasonReliabilityHead(dim, reason_dim, tail_indices=tail_indices)
        self.router = ParetoSafeRouter(dim, action_dim, reason_dim, action_residual_cap=action_residual_cap)

    def router_scale_for_epoch(self, epoch: int, router_start_epoch: int = 11, router_warmup_epochs: int = 5) -> float:
        if epoch < router_start_epoch:
            return 0.0 if epoch < 5 else 0.5
        return min(1.0, (epoch - router_start_epoch + 1) / max(1, router_warmup_epochs))

    def shared_parameters_for_budget(self):
        yield from self.shared_encoder.parameters()
        yield from self.ple.shared_1.parameters()
        yield from self.ple.shared_2.parameters()
        yield from self.pair_head.parameters()

    def forward(self, tokens: torch.Tensor, action_labels: torch.Tensor | None = None, reason_labels: torch.Tensor | None = None, epoch: int = 0) -> dict[str, torch.Tensor]:
        base = self.shared_encoder(tokens)
        experts = self.ple(base["action_tokens"], base["reason_tokens"], base["shared_context"])
        a_tokens = experts["action_tokens"]
        r_tokens = experts["reason_tokens"]
        a_action = self.action_head(a_tokens).squeeze(-1)
        r_reason = self.reason_head(r_tokens).squeeze(-1)
        a_reason_aux = self.action_aux_reason(r_tokens).squeeze(-1)
        r_action_aux = self.reason_aux_action(a_tokens).squeeze(-1)
        pair = self.pair_head(a_tokens, r_tokens, base["shared_context"])
        action_set = self.action_set_head(a_tokens)
        evidence = self.evidence_bag(tokens, r_tokens, reason_labels)
        action_agreement = torch.sigmoid(a_action).detach()
        reliability = self.reliability(
            r_tokens,
            r_reason,
            pair["pair_reason_support"],
            action_agreement,
            evidence["evidence_selected_score"].sigmoid().detach(),
            epoch=epoch,
            warmup_epochs=5,
        )
        router = self.router(
            a_action,
            r_reason,
            pair["pair_action_support"],
            pair["pair_reason_support"],
            action_set["action_set_logits"],
            reliability["reason_reliability"],
            base["shared_context"],
            self.router_scale_for_epoch(epoch),
        )
        return {
            **base,
            **experts,
            **pair,
            **action_set,
            **evidence,
            **reliability,
            **router,
            "base_action_logits": base["action_fused_logits"],
            "base_reason_logits": base["reason_logits"],
            "action_specialist_logits": a_action,
            "reason_specialist_logits": r_reason,
            "a_reason_aux_logits": a_reason_aux,
            "r_action_aux_logits": r_action_aux,
            "action_logits": router["final_action_logits"],
            "reason_logits": router["final_reason_logits"],
            "single_model_p3le_pair": tokens.new_tensor(1.0),
        }
