from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ZeroInitLowRankResidual(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: Tensor) -> Tensor:
        return self.up(torch.nn.functional.gelu(self.down(self.norm(value))))

    def parameter_scaled_residual(self, value: Tensor, scale: float) -> Tensor:
        """Preserve input gradients while applying a scale only to this block's parameters."""
        if not 0.0 <= float(scale) <= 1.0:
            raise ValueError("HECA shared gradient scale must lie in [0, 1]")

        def residual(input_value: Tensor, *, detach_parameters: bool) -> Tensor:
            norm_weight = self.norm.weight.detach() if detach_parameters else self.norm.weight
            norm_bias = self.norm.bias.detach() if detach_parameters else self.norm.bias
            down_weight = self.down.weight.detach() if detach_parameters else self.down.weight
            up_weight = self.up.weight.detach() if detach_parameters else self.up.weight
            normalized = F.layer_norm(
                input_value,
                self.norm.normalized_shape,
                norm_weight,
                norm_bias,
                self.norm.eps,
            )
            return F.linear(F.gelu(F.linear(normalized, down_weight)), up_weight)

        # The first path contributes the original gradient to label_nodes and
        # foundation. The second path contributes only parameter gradients.
        input_path = residual(value, detach_parameters=True)
        parameter_path = residual(value.detach(), detach_parameters=False)
        return input_path + float(scale) * (parameter_path - parameter_path.detach())


class HECASharedPrivateAdapters(nn.Module):
    """Zero-effect shared/action/reason adapters after label self-attention."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.shared_adapter = _ZeroInitLowRankResidual(dim, rank)
        self.action_private_adapter = _ZeroInitLowRankResidual(dim, rank)
        self.reason_private_adapter = _ZeroInitLowRankResidual(dim, rank)
        self.pu_private_head = nn.Linear(dim, 1)

    def forward(
        self,
        label_nodes: Tensor,
        *,
        shared_action_gradient_scale: float = 1.0,
        shared_reason_gradient_scale: float = 1.0,
    ) -> dict[str, Tensor]:
        if label_nodes.shape[1] != self.action_dim + self.reason_dim:
            raise ValueError("HECA adapters require action+reason label nodes")
        # Action/reason scales affect only shared-adapter parameters. The foundation
        # keeps its unweighted residual input gradient, matching the original policy.
        action_delta = self.shared_adapter.parameter_scaled_residual(
            label_nodes[:, : self.action_dim], shared_action_gradient_scale
        )
        reason_delta = self.shared_adapter.parameter_scaled_residual(
            label_nodes[:, self.action_dim :], shared_reason_gradient_scale
        )
        shared_delta = torch.cat((action_delta, reason_delta), dim=1)
        shared = label_nodes + shared_delta
        action = label_nodes[:, : self.action_dim] + action_delta
        reason = label_nodes[:, self.action_dim :] + reason_delta
        reason_private_delta = self.reason_private_adapter(reason)
        pu_private_nodes = (
            reason.detach()
            + self.reason_private_adapter(reason.detach())
        )
        return {
            "shared_nodes": shared,
            "action_nodes": action + self.action_private_adapter(action),
            "reason_nodes": reason + reason_private_delta,
            "reason_logits_pu_private": self.pu_private_head(
                pu_private_nodes
            ).squeeze(-1),
        }


class METERFactorMetaAdapters(nn.Module):
    """Per-factor low-rank adapters; only these expose a reason gradient bridge."""

    def __init__(self, factor_dim: int = 21, dim: int = 384, rank: int = 16) -> None:
        super().__init__()
        self.factor_dim = int(factor_dim)
        self.dim = int(dim)
        self.rank = int(rank)
        self.down = nn.Parameter(torch.empty(factor_dim, rank, dim))
        self.up = nn.Parameter(torch.zeros(factor_dim, dim, rank))
        nn.init.xavier_uniform_(self.down)

    def forward(self, core_tokens: Tensor, parameter_override: dict[str, Tensor] | None = None) -> Tensor:
        if core_tokens.ndim != 3 or core_tokens.shape[1:] != (self.factor_dim, self.dim):
            raise ValueError("Meta adapters require [B,factor_dim,dim] core tokens")
        parameter_override = parameter_override or {}
        down = parameter_override.get("down", self.down)
        up = parameter_override.get("up", self.up)
        if down.shape != self.down.shape or up.shape != self.up.shape:
            raise ValueError("Functional meta adapter override has an invalid shape")
        hidden = torch.einsum("bfd,frd->bfr", core_tokens.detach(), down)
        return torch.einsum("bfr,fdr->bfd", torch.nn.functional.gelu(hidden), up)

    def parameter_grad_norm(self) -> float:
        total = torch.zeros((), device=self.down.device)
        for parameter in self.parameters():
            if parameter.grad is not None:
                total = total + parameter.grad.detach().float().square().sum()
        return float(total.sqrt().cpu())
