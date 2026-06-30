from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NativeTextReasonPairMemory(nn.Module):
    def __init__(self, reason_dim: int = 21, capacity_per_reason: int = 256, tail_indices: list[int] | None = None, dim: int = 384) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.capacity_per_reason = capacity_per_reason
        self.tail_indices = tail_indices or [12, 9, 5, 14, 6, 11, 10, 13]
        self.proj = nn.Linear(dim, dim)
        self.register_buffer("pos_logits", torch.zeros(reason_dim, capacity_per_reason))
        self.register_buffer("neg_logits", torch.zeros(reason_dim, capacity_per_reason))
        self.register_buffer("pos_ptr", torch.zeros(reason_dim, dtype=torch.long))
        self.register_buffer("neg_ptr", torch.zeros(reason_dim, dtype=torch.long))
        self.register_buffer("pos_count", torch.zeros(reason_dim, dtype=torch.long))
        self.register_buffer("neg_count", torch.zeros(reason_dim, dtype=torch.long))

    def projection_parameters(self):
        return self.proj.parameters()

    @torch.no_grad()
    def enqueue(self, logits: torch.Tensor, targets: torch.Tensor, pu_state: dict) -> None:
        hard_neg = pu_state["hard_negative_mask"].bool()
        for r in range(self.reason_dim):
            pos_vals = logits[targets[:, r] > 0.5, r].detach()
            neg_vals = logits[hard_neg[:, r], r].detach()
            for vals, buf, ptr, cnt in ((pos_vals, self.pos_logits, self.pos_ptr, self.pos_count), (neg_vals, self.neg_logits, self.neg_ptr, self.neg_count)):
                if vals.numel() == 0:
                    continue
                take = min(vals.numel(), self.capacity_per_reason)
                idx = (torch.arange(take, device=buf.device) + ptr[r]) % self.capacity_per_reason
                buf[r, idx] = vals[:take].to(buf.device)
                ptr[r] = (ptr[r] + take) % self.capacity_per_reason
                cnt[r] = torch.clamp(cnt[r] + take, max=self.capacity_per_reason)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, pu_state: dict, epoch: int, main_loss: torch.Tensor, near_boundary_delta: float = 0.35, cap_ratio: float = 0.05) -> tuple[torch.Tensor, dict]:
        if epoch < 7:
            z = logits.sum() * 0.0
            return z, {"pair_count_total": 0, "zero_pair_count": int(logits.numel()), "memory_positive_coverage": 0, "memory_negative_coverage": 0, "cap_applied": False}
        losses = []
        pair_count = 0
        for r in range(self.reason_dim):
            pos_mask = targets[:, r] > 0.5
            neg_mask = pu_state["hard_negative_mask"][:, r] > 0.5
            if pos_mask.any() and neg_mask.any():
                up = logits[pos_mask, r].mean()
                un = logits[neg_mask, r].mean()
                losses.append(F.relu(0.2 - up + un))
                pair_count += 1
        raw = torch.stack(losses).mean() if losses else logits.sum() * 0.0
        cap = main_loss.detach() * cap_ratio
        capped = torch.minimum(raw, cap)
        self.enqueue(logits.detach(), targets.detach(), pu_state)
        return capped, {"pair_count_total": int(pair_count), "zero_pair_count": int(self.reason_dim - pair_count), "memory_positive_coverage": int((self.pos_count > 0).sum().detach().cpu()), "memory_negative_coverage": int((self.neg_count > 0).sum().detach().cpu()), "cap_applied": bool(raw.detach().cpu() > cap.detach().cpu())}
