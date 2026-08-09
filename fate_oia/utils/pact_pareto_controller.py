from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn


class PACTParetoController(nn.Module):
    """Held-out lookahead selector that never mutates formal model parameters."""

    def __init__(self, candidates=(0.0, 0.25, 0.5, 0.75), epsilon_action_audit=0.001,
                 license_ema=0.80, initial_license=0.0, action_best_tolerance=0.00005,
                 semantic_improvement_min=0.0) -> None:
        super().__init__()
        self.candidates = tuple(float(x) for x in candidates)
        self.epsilon_action_audit = float(epsilon_action_audit)
        self.license_ema = float(license_ema)
        self.action_best_tolerance = float(action_best_tolerance)
        self.semantic_improvement_min = float(semantic_improvement_min)
        self.register_buffer("semantic_share_license", torch.tensor(float(initial_license)))
        self.register_buffer("audit_rotation", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def select(self, candidate_measurements: dict[float, dict[str, float]]) -> dict:
        measurements = {
            float(candidate): {
                "action_loss": float(values["action_loss"]),
                "semantic_loss": float(values["semantic_loss"]),
            }
            for candidate, values in candidate_measurements.items()
        }
        baseline = measurements[0.0]
        safe = [value for value in self.candidates
                if measurements[value]["action_loss"] <= baseline["action_loss"] + self.epsilon_action_audit]
        best_action = min(measurements[value]["action_loss"] for value in safe)
        priority_safe = [value for value in safe
                         if measurements[value]["action_loss"] <= best_action + self.action_best_tolerance]
        useful = [value for value in priority_safe if value > 0 and
                  measurements[value]["semantic_loss"] <
                  baseline["semantic_loss"] - self.semantic_improvement_min]
        selected = min(useful, key=lambda value: (measurements[value]["semantic_loss"], value)) if useful else 0.0
        old = float(self.semantic_share_license)
        updated = self.license_ema * old + (1.0 - self.license_ema) * selected
        self.semantic_share_license.fill_(updated)
        self.audit_rotation.add_(1)
        return {"candidate_measurements": {str(k): v for k, v in measurements.items()},
                "candidate_action_losses": {str(k): v["action_loss"] for k, v in measurements.items()},
                "candidate_semantic_losses": {str(k): v["semantic_loss"] for k, v in measurements.items()},
                "action_best_loss": best_action, "action_priority_safe": priority_safe,
                "selected_lambda": selected, "license_before": old, "license_after": updated,
                "audit_rotation": int(self.audit_rotation)}

    def evaluate_candidates(self, evaluator: Callable[[float], dict[str, float]], model: nn.Module) -> dict:
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        measurements = {candidate: evaluator(candidate) for candidate in self.candidates}
        after = model.state_dict()
        changed = [key for key in before if not torch.equal(before[key], after[key])]
        if changed:
            model.load_state_dict(before)
            raise RuntimeError(f"Pareto candidate evaluator mutated formal model: {changed[:4]}")
        return self.select(measurements)

    def get_extra_state(self):
        return {"candidates": self.candidates, "epsilon": self.epsilon_action_audit, "ema": self.license_ema,
                "action_best_tolerance": self.action_best_tolerance,
                "semantic_improvement_min": self.semantic_improvement_min}

    def set_extra_state(self, state):
        self.candidates = tuple(state["candidates"])
        self.epsilon_action_audit = float(state["epsilon"])
        self.license_ema = float(state["ema"])
        self.action_best_tolerance = float(state.get("action_best_tolerance", 0.00005))
        self.semantic_improvement_min = float(state.get("semantic_improvement_min", 0.0))
