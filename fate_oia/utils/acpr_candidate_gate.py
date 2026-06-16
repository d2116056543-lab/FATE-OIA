from __future__ import annotations

import torch


class ACPRActionCandidateGate:
    def __init__(
        self,
        candidate_names,
        action_names=("forward", "stop", "left", "right"),
        min_delta_f1: float = 0.002,
        max_exp_drop: float = 0.005,
        gate_ema: float = 0.20,
        gate_max_all_high_increase: float = 0.02,
        gate_max_action_pred_rate_increase_abs: float = 0.08,
        gate_max_action_pred_rate_increase_rel: float = 1.5,
    ) -> None:
        self.candidate_names = list(candidate_names)
        self.action_names = list(action_names)
        self.min_delta_f1 = float(min_delta_f1)
        self.max_exp_drop = float(max_exp_drop)
        self.gate_ema = float(gate_ema)
        self.gate_max_all_high_increase = float(gate_max_all_high_increase)
        self.gate_max_action_pred_rate_increase_abs = float(gate_max_action_pred_rate_increase_abs)
        self.gate_max_action_pred_rate_increase_rel = float(gate_max_action_pred_rate_increase_rel)
        self.selected_candidate_id = torch.full((len(self.action_names),), -1, dtype=torch.long)
        self.selected_gate = torch.zeros(len(self.action_names), dtype=torch.float32)

    def state_dict(self) -> dict:
        return {
            "selected_candidate_id": self.selected_candidate_id.clone(),
            "selected_gate": self.selected_gate.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.selected_candidate_id.copy_(state["selected_candidate_id"].long())
        self.selected_gate.copy_(state["selected_gate"].float())

    def update_from_train_calib(
        self,
        candidate_metrics: dict,
        fallback_metrics: dict | None = None,
        exp_metrics_optional: dict | None = None,
    ) -> dict:
        fallback = fallback_metrics or candidate_metrics["fallback"]
        diagnostics = {
            "source": "train_calib_only",
            "candidate_names": self.candidate_names,
            "selected_candidate_id": [],
            "selected_gate": [],
        }
        fallback_act = float(fallback.get("Act_mF1", 0.0))
        fallback_all_high = float(fallback.get("all_high_rate", 0.0))
        fallback_rates = list(fallback.get("predicted_positive_rate_per_action", [0.0] * len(self.action_names)))
        exp_drop = float((exp_metrics_optional or {}).get("exp_drop", 0.0))
        for a, action_name in enumerate(self.action_names):
            best_name = "fallback"
            best_id = -1
            best_f1 = float(fallback["per_action_F1"][a])
            rejected = {}
            for idx, name in enumerate(self.candidate_names):
                m = candidate_metrics[name]
                cand_f1 = float(m["per_action_F1"][a])
                cand_rate = float(m["predicted_positive_rate_per_action"][a])
                rate_limit = max(float(fallback_rates[a]) * self.gate_max_action_pred_rate_increase_rel, float(fallback_rates[a]) + self.gate_max_action_pred_rate_increase_abs)
                reasons = []
                if cand_f1 - float(fallback["per_action_F1"][a]) < self.min_delta_f1:
                    reasons.append("delta_f1_below_min")
                if float(m.get("all_high_rate", 0.0)) > fallback_all_high + self.gate_max_all_high_increase:
                    reasons.append("all_high_increase")
                if cand_rate > rate_limit:
                    reasons.append("pred_rate_explosion")
                if float(m.get("Act_mF1", 0.0)) < fallback_act - 0.005:
                    reasons.append("overall_action_drop")
                if exp_drop > self.max_exp_drop:
                    reasons.append("exp_drop")
                if reasons:
                    rejected[name] = reasons
                    continue
                if cand_f1 > best_f1:
                    best_f1 = cand_f1
                    best_name = name
                    best_id = idx
            target_gate = 1.0 if best_id >= 0 else 0.0
            if best_id >= 0:
                if int(self.selected_candidate_id[a].item()) != best_id:
                    self.selected_candidate_id[a] = best_id
                    self.selected_gate[a] = self.gate_ema
                else:
                    self.selected_gate[a] = (1.0 - self.gate_ema) * self.selected_gate[a] + self.gate_ema * target_gate
            else:
                self.selected_candidate_id[a] = -1
                self.selected_gate[a] = (1.0 - self.gate_ema) * self.selected_gate[a]
                if float(self.selected_gate[a].item()) < 1e-6:
                    self.selected_gate[a] = 0.0
            delta = best_f1 - float(fallback["per_action_F1"][a])
            diagnostics[f"selected_candidate_{action_name}"] = best_name
            diagnostics[f"delta_f1_{action_name}"] = float(delta)
            diagnostics[f"fallback_f1_{action_name}"] = float(fallback["per_action_F1"][a])
            diagnostics[f"candidate_f1_{action_name}"] = float(best_f1)
            diagnostics[f"gate_{action_name}"] = float(self.selected_gate[a].item())
            diagnostics[f"rejected_{action_name}"] = rejected
        diagnostics["selected_candidate_id"] = [int(x) for x in self.selected_candidate_id.tolist()]
        diagnostics["selected_gate"] = [float(x) for x in self.selected_gate.tolist()]
        diagnostics["selected_gates_nonzero"] = any(x > 0 for x in diagnostics["selected_gate"])
        return diagnostics
