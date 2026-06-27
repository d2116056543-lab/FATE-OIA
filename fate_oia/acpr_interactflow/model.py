from __future__ import annotations

import torch
from torch import nn

from .decision_ledger import DecisionLedgerHead
from .dynamic_predicate_field import DynamicPredicateField
from .exp29_head import Exp29Head
from .interaction_flow import InteractionFlowReasoner
from .motion_path import MotionPathEncoder
from .nnpu_calalign import NNPUCalAlignHead
from .state_bank import ObjectiveEnvironmentStateBank
from .types import ACPRInteractFlowPPOutput
from .visual_encoder import InteractVisualEncoder


class ACPRInteractFlowPPModel(nn.Module):
    """Formal PSI model. It does not instantiate legacy OIA model or dataset classes."""

    def __init__(
        self,
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        predicate_config: str = "configs/acpr_interactflow_predicates.yaml",
        grammar_path: str = "configs/acpr_interactflow_state_grammar.yaml",
        exp29_names_path: str | None = None,
        dim: int = 384,
        action_dim: int = 4,
        use_mock_dino: bool = False,
    ) -> None:
        super().__init__()
        self.visual = InteractVisualEncoder(pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino, dim=dim)
        self.motion = MotionPathEncoder(dim=dim)
        self.predicates = DynamicPredicateField(predicate_config=predicate_config, dim=dim)
        self.state_bank = ObjectiveEnvironmentStateBank(grammar_path=grammar_path, dim=dim, num_predicates=48)
        self.flow = InteractionFlowReasoner(grammar_path=grammar_path, dim=dim, num_predicates=48, num_actions=action_dim)
        self.ledger = DecisionLedgerHead(dim=dim, num_actions=action_dim)
        self.exp29 = Exp29Head(dim=dim, exp_dim=29, label_names_path=exp29_names_path)
        self.calalign = NNPUCalAlignHead(action_dim=action_dim, exp_dim=29)

    def forward(self, input_frames: torch.Tensor, epoch: int = 0) -> ACPRInteractFlowPPOutput:
        visual = self.visual(input_frames)
        motion = self.motion(visual.fast_motion_tokens)
        predicates = self.predicates(visual.patch_tokens_by_layer)
        state_bank = self.state_bank(predicates.predicate_tokens, predicates.predicate_probs)
        flow = self.flow(predicates.predicate_tokens, predicates.predicate_probs, motion["motion_token"])
        visual_token = visual.anchor_tokens[:, -1]
        predicate_token = predicates.predicate_tokens.mean(1) + state_bank["state_tokens"].mean(1)
        ledger = self.ledger(visual_token, motion["motion_token"], predicate_token, flow.factor_tokens, flow.flow_edges)
        exp29 = self.exp29(flow.factor_tokens, predicates.predicate_tokens)
        cal = self.calalign(ledger.final_logits, exp29.logits)
        aux = {
            "action_logits_calibrated": cal["action_logits_calibrated"],
            "exp29_logits_calibrated": cal["exp29_logits_calibrated"],
            "state_group_logits": state_bank["state_group_logits"],
            "state_layer_weights": state_bank["state_layer_weights"],
            "state_attention": state_bank["state_attention"],
            "state_stats": state_bank["state_stats"],
            "motion_logits": motion["motion_logits"],
            "epoch": epoch,
        }
        return ACPRInteractFlowPPOutput(
            action_logits=ledger.final_logits,
            action_probs=torch.softmax(ledger.final_logits, dim=-1),
            exp29_logits=exp29.logits,
            exp29_probs=exp29.probs,
            visual=visual,
            predicates=predicates,
            flow=flow,
            ledger=ledger,
            exp29=exp29,
            aux=aux,
        )
