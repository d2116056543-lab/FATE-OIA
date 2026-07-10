from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .mosaic_sparse_label_decoder import MOSAICSparseLabelDecoder


class MOSAICActionDecoder(nn.Module):
    def __init__(
        self,
        num_states: int,
        *,
        dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
    ) -> None:
        super().__init__()
        if type(num_states) is not int or num_states <= 0:
            raise ValueError("action decoder requires a positive state count")
        self.num_states = num_states
        self.dim = dim
        self.visual_decoder = MOSAICSparseLabelDecoder(
            4,
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.state_embeddings = nn.Parameter(torch.randn(num_states, dim) * 0.02)
        self.state_key = nn.Linear(dim, dim, bias=False)
        self.state_value = nn.Linear(dim, dim, bias=False)
        self.action_state_query = nn.Linear(dim, dim, bias=False)
        self.state_classifier_weight = nn.Parameter(torch.empty(4, dim))
        self.state_classifier_bias = nn.Parameter(torch.zeros(4))
        self.gate_weight = nn.Parameter(torch.empty(4, dim))
        self.gate_bias = nn.Parameter(torch.zeros(4))
        nn.init.xavier_uniform_(self.state_classifier_weight)
        nn.init.xavier_uniform_(self.gate_weight)

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        state_prob: torch.Tensor,
        state_uncertainty: torch.Tensor,
        *,
        state_gate_cap: float,
    ) -> dict[str, torch.Tensor]:
        if type(state_gate_cap) not in {int, float} or not 0.0 <= float(state_gate_cap) <= 0.25:
            raise ValueError("state_gate_cap must be in [0,0.25]")
        visual = self.visual_decoder(pyramid)
        action_nodes = visual["label_nodes"]
        batch_size = action_nodes.shape[0]
        expected_state_shape = (batch_size, self.num_states)
        if tuple(state_prob.shape) != expected_state_shape or tuple(state_uncertainty.shape) != expected_state_shape:
            raise ValueError("action decoder state shape contract is invalid")

        state_tokens = self.state_embeddings.unsqueeze(0) * state_prob.unsqueeze(-1)
        keys = F.normalize(self.state_key(state_tokens), dim=-1, eps=1e-6)
        queries = F.normalize(self.action_state_query(action_nodes), dim=-1, eps=1e-6)
        state_scores = torch.einsum("bad,bkd->bak", queries, keys) / math.sqrt(self.dim)
        state_attention = torch.softmax(state_scores, dim=-1)
        state_context = torch.einsum("bak,bkd->bad", state_attention, self.state_value(state_tokens))
        state_logits = torch.einsum("bad,ad->ba", state_context, self.state_classifier_weight) + self.state_classifier_bias
        state_confidence = torch.einsum(
            "bak,bk->ba", state_attention, 1.0 - state_uncertainty.clamp(0.0, 1.0)
        )
        learned_gate = torch.sigmoid(torch.einsum("bad,ad->ba", action_nodes, self.gate_weight) + self.gate_bias)
        state_gate = float(state_gate_cap) * learned_gate * state_confidence
        visual_logits = visual["label_logits"]
        raw_logits = visual_logits + state_gate * state_logits
        return {
            "action_visual_logits": visual_logits,
            "action_state_logits": state_logits,
            "action_state_gate": state_gate,
            "action_logits_raw": raw_logits,
            "action_nodes": action_nodes,
            "action_state_attention": state_attention,
        }
