from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class PACTParetoController(nn.Module):
    """Held-out lookahead selector that never mutates formal model parameters."""

    def __init__(self, candidates=(0.0, 0.25, 0.5, 0.75), epsilon_action_audit=0.001,
                 license_ema=0.80, initial_license=0.0) -> None:
        super().__init__()
        self.candidates = tuple(float(x) for x in candidates)
        self.epsilon_action_audit = float(epsilon_action_audit)
        self.license_ema = float(license_ema)
        self.register_buffer("semantic_share_license", torch.tensor(float(initial_license)))
        self.register_buffer("audit_rotation", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def select(self, candidate_action_losses: dict[float, float]) -> dict:
        baseline = float(candidate_action_losses[0.0])
        safe = [value for value in self.candidates
                if float(candidate_action_losses[value]) <= baseline + self.epsilon_action_audit]
        selected = max(safe) if safe else 0.0
        old = float(self.semantic_share_license)
        updated = self.license_ema * old + (1.0 - self.license_ema) * selected
        self.semantic_share_license.fill_(updated)
        self.audit_rotation.add_(1)
        return {"candidate_action_losses": {str(k): float(v) for k, v in candidate_action_losses.items()},
                "selected_lambda": selected, "license_before": old, "license_after": updated,
                "audit_rotation": int(self.audit_rotation)}

    def evaluate_candidates(self, evaluator: Callable[[float], float], model: nn.Module) -> dict:
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        losses = {candidate: float(evaluator(candidate)) for candidate in self.candidates}
        after = model.state_dict()
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        if changed:
            model.load_state_dict(before)
            raise RuntimeError(f"Pareto candidate evaluator mutated formal model: {changed[:4]}")
        return self.select(losses)

    def get_extra_state(self):
        return {"candidates": self.candidates, "epsilon": self.epsilon_action_audit, "ema": self.license_ema}

    def set_extra_state(self, state):
        self.candidates = tuple(state["candidates"])
        self.epsilon_action_audit = float(state["epsilon"])
        self.license_ema = float(state["ema"])
