from __future__ import annotations

import torch
from torch import nn


class ACPRFusionLiteGate(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_predicates: int = 0,
        max_delta: float = 0.15,
        gate_min: float = 0.10,
        gate_max: float = 0.90,
        use_predicate_context: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_predicates = int(num_predicates)
        self.max_delta = float(max_delta)
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        self.use_predicate_context = bool(use_predicate_context)
        self.predicate_proj = nn.Linear(max(self.num_predicates, 1), dim)
        self.scalar_proj = nn.Linear(4, dim)
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        final = self.delta_mlp[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _predicate_context(
        self,
        predicate_probs: torch.Tensor | None,
        action_predicate_mask: torch.Tensor | None,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if (not self.use_predicate_context) or predicate_probs is None or action_predicate_mask is None or self.num_predicates <= 0:
            return torch.zeros(batch, self.action_dim, self.dim, device=device, dtype=dtype)
        p = predicate_probs.to(device=device, dtype=dtype)
        mask = action_predicate_mask.to(device=device, dtype=dtype)
        if p.shape[-1] != self.num_predicates:
            if p.shape[-1] > self.num_predicates:
                p = p[..., : self.num_predicates]
            else:
                p = torch.cat([p, p.new_zeros(p.shape[0], self.num_predicates - p.shape[-1])], dim=-1)
        if mask.shape[-1] != self.num_predicates:
            if mask.shape[-1] > self.num_predicates:
                mask = mask[..., : self.num_predicates]
            else:
                mask = torch.cat([mask, mask.new_zeros(mask.shape[0], self.num_predicates - mask.shape[-1])], dim=-1)
        masked_pred = p[:, None, :] * mask[None, :, :]
        return self.predicate_proj(masked_pred)

    def forward(
        self,
        action_nodes: torch.Tensor,
        reason_nodes: torch.Tensor,
        predicate_probs: torch.Tensor | None,
        action_visual_logits: torch.Tensor,
        action_reason_logits: torch.Tensor,
        old_gate: torch.Tensor,
        action_reason_mask: torch.Tensor,
        action_predicate_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        b, a, _ = action_nodes.shape
        if a != self.action_dim or reason_nodes.shape[1] != self.reason_dim:
            raise ValueError("FusionLite received unexpected action/reason node shapes")
        reason_context = torch.einsum("ar,brd->bad", action_reason_mask.to(reason_nodes.device, reason_nodes.dtype), reason_nodes)
        predicate_context = self._predicate_context(predicate_probs, action_predicate_mask, b, action_nodes.device, action_nodes.dtype)
        scalar = torch.stack([action_visual_logits, action_reason_logits, action_reason_logits - action_visual_logits, old_gate], dim=-1)
        scalar_embed = self.scalar_proj(scalar.to(action_nodes.dtype))
        x = torch.cat([action_nodes, reason_context, predicate_context, scalar_embed], dim=-1)
        raw_delta = self.delta_mlp(x).squeeze(-1)
        delta = self.max_delta * torch.tanh(raw_delta)
        new_gate = (old_gate + delta).clamp(self.gate_min, self.gate_max)
        action_logits = new_gate * action_visual_logits + (1.0 - new_gate) * action_reason_logits
        return {
            "action_logits_fusionlite": action_logits,
            "fusionlite_gate": new_gate,
            "fusionlite_delta_gate": delta,
            "fusionlite_reason_context": reason_context,
            "fusionlite_predicate_context": predicate_context,
            "fusionlite_delta_abs_mean": delta.abs().mean(),
            "fusionlite_gate_mean": new_gate.mean(),
            "fusionlite_old_gate_mean": old_gate.mean(),
        }
