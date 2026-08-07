from __future__ import annotations

import torch
from torch import Tensor


class AIECertCalibrationGuard:
    def __init__(self, labels: int, ema_decay=0.80, max_step=0.05):
        self.ema_decay, self.max_step = float(ema_decay), float(max_step)
        self.accepted = torch.full((labels,), 0.5)

    def propose(self, candidate: Tensor, raw_joint: float, raw_action: float, deploy_joint: float,
                deploy_action: float) -> dict:
        current = self.accepted.to(candidate)
        clipped = current + (candidate - current).clamp(-self.max_step, self.max_step)
        ema = self.ema_decay * current + (1.0 - self.ema_decay) * clipped
        accept = deploy_joint >= raw_joint - 0.001 and deploy_action >= raw_action - 0.002
        if accept:
            self.accepted = ema.detach().cpu()
        return {"candidate_threshold": candidate.detach().cpu(), "accepted_threshold": self.accepted.clone(),
                "accepted": accept, "reject_reason": "" if accept else "train_calib_guard"}

    def state_dict(self):
        return {"accepted": self.accepted, "ema_decay": self.ema_decay, "max_step": self.max_step}

    def load_state_dict(self, state):
        self.accepted = state["accepted"].clone()
        self.ema_decay, self.max_step = float(state["ema_decay"]), float(state["max_step"])
