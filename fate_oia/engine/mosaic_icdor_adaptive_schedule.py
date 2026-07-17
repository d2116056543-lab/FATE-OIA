from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fate_oia.engine.mosaic_icdor_schedule import ICDORPhase

STATES = ("FOUNDATION", "DUAL_REASON_SHADOW", "SAFE_JOINT", "CONSOLIDATION")


@dataclass(frozen=True)
class ICDORStatePolicy:
    route_mode: str
    action_rank_weight: float
    reason_rank_weight: float
    write_provisional_certificate: bool = False
    freeze_factor_and_prototypes: bool = False
    freeze_certificate: bool = False
    freeze_edge_admission: bool = False
    freeze_propensity: bool = False
    enable_interventions: bool = False
    enable_hidden_audit: bool = False
    enable_pareto: bool = False
    route_is_final: bool = False
    decoder_router_lr_scale: float = 1.0
    route_cap_mutable: bool = True
    allow_new_losses: bool = True
    pu_enabled: bool = True


POLICIES = {
    # Regime A: all learning routes are active, but action shadow is not final.
    # Continuous visual credibility grants learning access. A discrete
    # certificate is a deployment claim and is therefore never built merely
    # to unlock FOUNDATION or shadow learning.
    "FOUNDATION": ICDORStatePolicy("shadow", 0.10, 0.05),
    "DUAL_REASON_SHADOW": ICDORStatePolicy(
        "shadow", 0.0, 0.0, freeze_factor_and_prototypes=True, freeze_certificate=True,
        enable_interventions=True, enable_hidden_audit=True,
    ),
    "SAFE_JOINT": ICDORStatePolicy(
        "admitted", 1.0, 1.0, freeze_factor_and_prototypes=True, freeze_certificate=True,
        freeze_edge_admission=True, enable_interventions=True, enable_hidden_audit=True,
        enable_pareto=True, route_is_final=True,
    ),
    "CONSOLIDATION": ICDORStatePolicy(
        "admitted", 1.0, 1.0, freeze_factor_and_prototypes=True, freeze_certificate=True,
        freeze_edge_admission=True, freeze_propensity=True, enable_pareto=True, route_is_final=True,
        decoder_router_lr_scale=0.2, route_cap_mutable=False, allow_new_losses=False,
    ),
}


class ICDORAdaptiveSchedule:
    """Continuous-access train-audit/train-calib state machine; test is forbidden."""

    LIMITS = {
        "FOUNDATION": (3, 6),
        "DUAL_REASON_SHADOW": (2, 5),
        "SAFE_JOINT": (4, 8),
        "CONSOLIDATION": (1, 2),
    }

    def __init__(self, *, pilot: bool) -> None:
        self.pilot = bool(pilot)
        self.state = "FOUNDATION"
        self.state_epochs = 0
        self.consecutive_ready = 0
        self.no_improvement_epochs = 0
        self.failed_closed = False
        self.failure_reason: str | None = None
        self.full_train_eligible = False
        self.safe_joint_epochs = 0
        self.best_train_audit_joint: float | None = None
        self.safe_joint_entry_exp_map: float | None = None
        self.last_readiness: dict[str, dict[str, Any]] = {}
        self.history: list[dict[str, Any]] = []
        self.pu_enabled = True
        self.pu_disable_reason: str | None = None

    def policy(self) -> ICDORStatePolicy:
        return POLICIES[self.state]

    def phase(self) -> ICDORPhase:
        """Translate the current adaptive state into the trainer's loss firewall."""
        policy = self.policy()
        return ICDORPhase(
            name=f"adaptive_{self.state.lower()}",
            route_mode=policy.route_mode,
            latent_enabled=True,
            enable_factor_losses=True,
            enable_posterior_ranking=policy.reason_rank_weight > 0.0,
            enable_pareto=policy.enable_pareto,
            freeze_factor_branch=policy.freeze_factor_and_prototypes,
            pu_enabled=self.pu_enabled,
        )

    def set_pu_enabled(self, enabled: bool, *, reason: str | None = None) -> None:
        self.pu_enabled = bool(enabled)
        self.pu_disable_reason = None if enabled else (reason or "disabled_by_policy")

    def record_epoch_execution(self) -> None:
        """Track the state that actually controlled an optimizer epoch."""
        if self.state == "SAFE_JOINT":
            self.safe_joint_epochs += 1

    def record_train_audit_reference(
        self, *, joint: float, exp_map: float, entered_safe_joint: bool
    ) -> None:
        """Persist train-audit comparison anchors across checkpoint resume."""
        joint_value = float(joint)
        exp_value = float(exp_map)
        if not self._finite(joint_value) or not self._finite(exp_value):
            raise ValueError("train-audit reference metrics must be finite")
        self.best_train_audit_joint = (
            joint_value if self.best_train_audit_joint is None
            else max(self.best_train_audit_joint, joint_value)
        )
        if entered_safe_joint:
            self.safe_joint_entry_exp_map = exp_value

    def fail_closed(self, reason: str) -> None:
        if not reason:
            raise ValueError("adaptive schedule failure reason is required")
        self.failed_closed = True
        self.failure_reason = reason

    @staticmethod
    def _finite(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return number == number and abs(number) != float("inf")

    def _foundation_ready(self, metrics: dict[str, Any]) -> bool:
        cV_count = int(metrics.get("observable_cV_gt_030", 0))
        return (
            metrics.get("continuous_credibility_available") is True
            and cV_count >= 6
            and metrics.get("diagnostics_finite") is True
        )

    def _shadow_ready(self, metrics: dict[str, Any]) -> bool:
        action_ap = metrics.get("action_shadow_ap_delta", [])
        true_edges = metrics.get("true_edge_count_per_action", [])
        minimum_actions = 2 if self.pilot else 3
        hidden_margin = 0.01 if self.pilot else 0.02
        return (
            float(metrics.get("exp_map_delta_vs_visual", -1.0)) >= -0.005
            and metrics.get("factor_shuffle_degrades_reason") is True
            and float(metrics.get("hidden_recovery_margin", -1.0)) >= hidden_margin
            and 0.01 <= float(metrics.get("route_strength_ratio", -1.0)) <= 0.15
            and sum(float(value) >= -0.002 for value in action_ap) >= minimum_actions
            and len(true_edges) == 4 and sum(int(value) >= 1 for value in true_edges) >= minimum_actions
            and metrics.get("disallowed_route_invariance") is True
        )

    def _safe_ready(self, metrics: dict[str, Any]) -> bool:
        action_ap = metrics.get("action_route_ap_delta", [])
        return (
            len(action_ap) == 4 and min(float(value) for value in action_ap) >= -0.002
            and 0.02 <= float(metrics.get("route_strength_ratio", -1.0)) <= 0.15
            and float(metrics.get("pareto_violation_rate", 1.0)) < 0.05
            and float(metrics.get("exp_map_delta_vs_entry", -1.0)) >= 0.0
            and float(metrics.get("tet_lcb95", 0.0)) > 0.0
            and float(metrics.get("tes_lcb95", 0.0)) > 0.0
            and float(metrics.get("cca", 0.0)) >= 0.60
        )

    def _ready(self, metrics: dict[str, Any]) -> bool:
        if self.state == "FOUNDATION":
            return self._foundation_ready(metrics)
        if self.state == "DUAL_REASON_SHADOW":
            return self._shadow_ready(metrics)
        if self.state == "SAFE_JOINT":
            return self._safe_ready(metrics)
        return True

    @staticmethod
    def _split_ready(metrics: dict[str, Any], expected_split: str) -> bool:
        return (
            metrics.get("source_split") == expected_split
            and metrics.get("finite") is True
        )

    def _transition(self) -> None:
        index = STATES.index(self.state)
        if index + 1 >= len(STATES):
            self.full_train_eligible = True
            return
        self.state = STATES[index + 1]
        self.state_epochs = 0
        self.consecutive_ready = 0
        self.no_improvement_epochs = 0

    def update(
        self,
        *,
        epoch: int,
        train_audit_metrics: dict[str, Any],
        train_calib_metrics: dict[str, Any],
        train_core_metrics: dict[str, Any] | None = None,
        test_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if test_metrics is not None:
            raise ValueError("test metrics are forbidden in the adaptive schedule")
        readiness = {
            "train_core": dict(train_core_metrics or {}),
            "train_audit": dict(train_audit_metrics),
            "train_calib": dict(train_calib_metrics),
        }
        for split, metrics in readiness.items():
            source_split = metrics.get("source_split")
            if source_split is not None and source_split != split:
                raise ValueError(f"adaptive schedule {split} readiness has invalid source {source_split!r}")
        self.last_readiness = readiness
        if self.failed_closed:
            return self.state_dict()
        previous = self.state
        state_epochs_before = self.state_epochs
        self.state_epochs += 1
        ready = (
            self._ready(train_audit_metrics)
            and self._split_ready(readiness["train_core"], "train_core")
            and self._split_ready(readiness["train_calib"], "train_calib")
        )
        self.consecutive_ready = self.consecutive_ready + 1 if ready else 0
        minimum, maximum = self.LIMITS[self.state]

        if self.state == "SAFE_JOINT":
            improved = bool(train_audit_metrics.get("train_audit_improved", False))
            self.no_improvement_epochs = 0 if improved else self.no_improvement_epochs + 1
            if self.state_epochs >= minimum and self.no_improvement_epochs >= 2:
                self._transition()
        elif self.state == "CONSOLIDATION":
            if self.state_epochs >= minimum:
                self.full_train_eligible = True
        elif self.state_epochs >= minimum and self.consecutive_ready >= 2:
            self._transition()

        # Missing audit evidence may delay a transition, but never removes
        # learning access.  Discrete certificates are only used by admission.
        row = {
            "epoch": int(epoch),
            "state_before": previous,
            "state_after": self.state,
            "state_epochs_before": state_epochs_before,
            "state_epochs_after": self.state_epochs,
            "ready": ready,
            "failed_closed": self.failed_closed,
            "readiness": readiness,
            "policy": asdict(self.policy()),
        }
        self.history.append(row)
        return row

    def state_dict(self) -> dict[str, Any]:
        return {
            "pilot": self.pilot,
            "state": self.state,
            "state_epochs": self.state_epochs,
            "consecutive_ready": self.consecutive_ready,
            "no_improvement_epochs": self.no_improvement_epochs,
            "failed_closed": self.failed_closed,
            "failure_reason": self.failure_reason,
            "full_train_eligible": self.full_train_eligible,
            "safe_joint_epochs": self.safe_joint_epochs,
            "best_train_audit_joint": self.best_train_audit_joint,
            "safe_joint_entry_exp_map": self.safe_joint_entry_exp_map,
            "last_readiness": dict(self.last_readiness),
            "history": list(self.history),
            "pu_enabled": self.pu_enabled,
            "pu_disable_reason": self.pu_disable_reason,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("state") not in STATES or bool(payload.get("pilot")) != self.pilot:
            raise ValueError("adaptive schedule resume state is incompatible")
        for key in (
            "state", "state_epochs", "consecutive_ready", "no_improvement_epochs",
            "failed_closed", "failure_reason", "full_train_eligible", "safe_joint_epochs",
            "best_train_audit_joint", "safe_joint_entry_exp_map", "last_readiness", "history",
        ):
            setattr(self, key, payload[key])
        self.pu_enabled = bool(payload.get("pu_enabled", True))
        self.pu_disable_reason = payload.get("pu_disable_reason")
