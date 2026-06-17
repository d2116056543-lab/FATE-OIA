from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import yaml


class ACPRTriadicMediator(nn.Module):
    # PMT-S invariant: predicate-only action delta is impossible; action deltas require action-reason-predicate support.
    def __init__(self, action_dim=4, reason_dim=21, num_predicates=32, rank=8, max_action_delta=0.10, grammar_path="configs/acpr_reason_predicate_grammar.yaml", action_predicate_grammar_path="configs/acpr_pmt_action_predicate_grammar.yaml"):
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_predicates = int(num_predicates)
        self.max_action_delta = float(max_action_delta)
        self.action_reason_weight = nn.Parameter(torch.zeros(action_dim, reason_dim))
        self.delta_scale = nn.Parameter(torch.tensor(0.0))
        self.A = nn.Parameter(torch.randn(action_dim, rank) * 0.01)
        self.R = nn.Parameter(torch.randn(reason_dim, rank) * 0.01)
        self.P = nn.Parameter(torch.randn(num_predicates, rank) * 0.01)
        ar = torch.ones(action_dim, reason_dim)
        ap = torch.ones(action_dim, num_predicates)
        rp = torch.zeros(reason_dim, num_predicates)
        rc = torch.zeros(reason_dim, num_predicates)
        if Path(action_predicate_grammar_path).exists():
            data = yaml.safe_load(Path(action_predicate_grammar_path).read_text(encoding="utf-8")) or {}
            for item in data.get("actions", {}).values():
                aid = int(item.get("id", 0))
                ar[aid].zero_()
                for rid in item.get("compatible_reasons", []):
                    if 0 <= int(rid) < reason_dim:
                        ar[aid, int(rid)] = 1
                # We cannot assume exact predicate indices from names here; keep
                # all-one action-predicate mask but real names are preserved in export.
        for r in range(reason_dim):
            rp[r, r % num_predicates] = 1.0
            rp[r, (r * 7 + 3) % num_predicates] = 1.0
            rc[r, (r * 5 + 1) % num_predicates] = 1.0
        self.register_buffer("action_reason_compat_mask", ar)
        self.register_buffer("action_predicate_support_mask", ap)
        self.register_buffer("reason_predicate_positive_mask", rp)
        self.register_buffer("reason_predicate_contradictory_mask", rc)

    def forward(self, action_visual_logits, action_reason_logits, reason_logits, predicate_probs, predicate_tokens=None):
        e = predicate_probs
        rp = (e @ self.reason_predicate_positive_mask.t()) / self.reason_predicate_positive_mask.sum(-1).clamp_min(1.0)
        ap = (e @ self.action_predicate_support_mask.t()) / self.action_predicate_support_mask.sum(-1).clamp_min(1.0)
        reason_conf = torch.sigmoid(reason_logits)
        support = self.action_reason_compat_mask.view(1, self.action_dim, self.reason_dim) * rp.unsqueeze(1) * ap.unsqueeze(-1)
        cp = torch.einsum("ak,rk,pk,bp->bar", self.A, self.R, self.P, e) / max(float(self.A.shape[-1]), 1.0)
        support = (support + 0.01 * cp).clamp(0.0, 1.0)
        # Zero-init is controlled by delta_scale. The low-rank compatibility term keeps
        # the mediator learnable/observable once the gate opens without changing init output.
        effective_weight = self.action_reason_weight.view(1, self.action_dim, self.reason_dim) + cp
        raw_delta = (reason_conf.unsqueeze(1) * support * effective_weight).sum(-1)
        raw_delta = raw_delta + 0.01 * (reason_conf.unsqueeze(1) * support).sum(-1)
        delta = self.max_action_delta * torch.tanh(self.delta_scale * raw_delta)
        top_chain = support.detach().flatten(1).topk(k=min(8, support.shape[1] * support.shape[2]), dim=1).indices
        stats = {"triadic_delta_abs_mean": float(delta.detach().abs().mean().cpu()), "triadic_support_mean": float(support.detach().mean().cpu())}
        return {
            "action_reason_logits_triadic": action_reason_logits + delta,
            "triadic_action_delta": delta,
            "triadic_chain_score": support,
            "triadic_reason_support": support,
            "triadic_predicate_support": self.action_predicate_support_mask.view(1, self.action_dim, self.num_predicates) * e.unsqueeze(1),
            "triadic_top_chain_indices": top_chain,
            "triadic_stats": stats,
        }
