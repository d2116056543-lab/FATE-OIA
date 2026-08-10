from __future__ import annotations

import torch
from torch import Tensor


class VETRACounterfactualAudit:
    """Conditional factor replacement in projected VETRA value space; never reruns DINO."""

    @staticmethod
    def _donor_distances(values: Tensor, selected: int, null_index: int) -> Tensor:
        distance = torch.linalg.vector_norm(values - values[selected], dim=-1)
        distance[selected] = torch.inf
        distance[null_index] = torch.inf
        return distance

    @staticmethod
    def _rerun_role(model, context: Tensor, route: Tensor, values: Tensor, selected: int,
                    donor: int, role: str, action: int) -> Tensor:
        modified = context + route[selected] * (values[donor] - values[selected])
        modified = model.transport.transport_norm(modified)
        head = model.transport.support_head if role == "support" else model.transport.counter_head
        return (modified * head[action]).sum()

    @torch.no_grad()
    def run(self, model, output: dict, action_target: Tensor, max_samples: int = 64) -> dict:
        cases = []; sample_count = min(int(max_samples), action_target.shape[0])
        for sample in range(sample_count):
            for action in range(4):
                sign = 2 * action_target[sample, action] - 1
                original = output["vetra_action_delta_unscaled"][sample, action]
                role_rows = []
                for role in ("support", "counter"):
                    route = output[f"{role}_route"][sample, action]
                    values = output[f"{role}_factor_values"][sample]
                    selected = int(route[:-1].argmax())
                    distance = self._donor_distances(values, selected, values.shape[0] - 1)
                    donors = distance.topk(k=min(2, max(distance.numel()-2, 1)), largest=False).indices
                    modified_scores = [self._rerun_role(model, output[f"{role}_context"][sample, action], route,
                                                        values, selected, int(donor), role, action) for donor in donors]
                    role_rows.append((role, selected, donors, modified_scores, distance[donors]))
                support_modified = role_rows[0][3][0]
                counter_modified = role_rows[1][3][0]
                cap = model.transport.correction_cap
                modified = cap * torch.tanh((support_modified - counter_modified) / cap)
                effect = sign * (original - modified)
                cases.append({"sample": sample, "action": action, "target_sign": float(sign),
                              "selected_control_effect": float(effect),
                              "support_selected": role_rows[0][1], "counter_selected": role_rows[1][1],
                              "support_donor_distance": float(role_rows[0][4][0]),
                              "counter_donor_distance": float(role_rows[1][4][0]),
                              "dino_rerun_count": 0})
        per_action = {str(a): {"count": sum(row["action"] == a for row in cases),
                               "effect_mean": (sum(row["selected_control_effect"] for row in cases if row["action"] == a)
                                               / max(sum(row["action"] == a for row in cases), 1))}
                      for a in range(4)}
        return {"cases": cases, "per_action": per_action, "valid_coverage": len(cases), "dino_rerun_count": 0}
