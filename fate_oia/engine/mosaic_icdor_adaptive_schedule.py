from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from fate_oia.engine.mosaic_icdor_schedule import ICDORPhase

STATES = ("JOINT_SHADOW", "ADMISSION_CONSOLIDATION")


@dataclass(frozen=True)
class ICDORStatePolicy:
    route_mode: str
    action_rank_weight: float
    reason_rank_weight: float
    reason_observed_rank_weight: float
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
    # V5 trains every representation owner from epoch zero.  Action remains
    # visual-only in this regime, while shadow routes receive cheap target
    # utility interventions every epoch and full audits every second epoch.
    "JOINT_SHADOW": ICDORStatePolicy(
        "shadow", 0.10, 0.05, 0.05,
        enable_interventions=True, enable_hidden_audit=True,
    ),
    # Deployment is an edge-level decision after representation learning; a
    # failed edge leaves only that action visual-only rather than blocking all
    # four actions behind a global factor credibility threshold.
    "ADMISSION_CONSOLIDATION": ICDORStatePolicy(
        "admitted", 0.10, 0.05, 0.05,
        freeze_certificate=True, freeze_edge_admission=True, freeze_propensity=True,
        enable_interventions=True, enable_hidden_audit=True, enable_pareto=True,
        route_is_final=True, decoder_router_lr_scale=0.2, route_cap_mutable=False,
        allow_new_losses=False,
    ),
}


class ICDORAdaptiveSchedule:
    """Continuous-access train-audit/train-calib state machine; test is forbidden."""

    LIMITS = {"JOINT_SHADOW": (8, 12), "ADMISSION_CONSOLIDATION": (2, 2)}

    def __init__(self, *, pilot: bool) -> None:
        self.pilot = bool(pilot)
        self.state = "JOINT_SHADOW"
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
        # PU is now a label-wise enhancement selected by posterior diagnostics.
        # Preserve fields for checkpoint compatibility, but never use one
        # global margin to disable latent semantic learning.
        self.pu_enabled = True
        self.pu_disable_reason: str | None = None

    def policy(self) -> ICDORStatePolicy:
        return POLICIES[self.state]

    def online_target_probe_due(self, *, epoch: int) -> bool:
        """Run the bounded target-utility probe from the first epoch onward."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        return self.policy().enable_interventions

    def full_target_audit_due(self, *, epoch: int, every_epochs: int = 2) -> bool:
        """Run bootstrap-quality target interventions on a fixed two-epoch cadence."""
        if epoch < 0 or every_epochs <= 0:
            raise ValueError("epoch must be non-negative and every_epochs positive")
        return self.policy().enable_interventions and epoch % every_epochs == 0

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
            pu_enabled=True,
        )

    def set_pu_enabled(self, enabled: bool, *, reason: str | None = None) -> None:
        # V5 intentionally ignores global PU closure. Store only a diagnostic
        # note; ``phase()`` still enables the latent core and per-label gate.
        self.pu_enabled = True
        self.pu_disable_reason = None if enabled else (reason or "labelwise_gate_required")

    def record_epoch_execution(self) -> None:
        """Track the state that actually controlled an optimizer epoch."""
        if self.state == "ADMISSION_CONSOLIDATION":
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

    def _ready(self, metrics: dict[str, Any]) -> bool:
        # No cV-count or global-PU opening gate remains.  Transition only
        # after a representation audit has produced finite edge evidence; the
        # admission builder itself decides each action independently.
        return metrics.get("diagnostics_finite") is True and metrics.get("final_action_visual_exact") is True

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
        previous_train_audit = self.last_readiness.get("train_audit", {})
        readiness = {
            "train_core": dict(train_core_metrics or {}),
            "train_audit": dict(train_audit_metrics),
            "train_calib": dict(train_calib_metrics),
        }
        for split, metrics in readiness.items():
            source_split = metrics.get("source_split")
            if source_split is not None and source_split != split:
                raise ValueError(f"adaptive schedule {split} readiness has invalid source {source_split!r}")
        current_audit = readiness["train_audit"]
        for metric_name, stable_name in (
            ("direct_action_map", "direct_action_map_stable_or_improving"),
            ("direct_reason_map", "direct_reason_map_stable_or_improving"),
        ):
            current = current_audit.get(metric_name)
            previous = previous_train_audit.get(metric_name)
            if not self._finite(current):
                current_audit[stable_name] = False
            elif not self._finite(previous):
                current_audit[stable_name] = True
            else:
                current_audit[stable_name] = float(current) >= float(previous) - 0.002
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

        if self.state == "ADMISSION_CONSOLIDATION":
            if self.state_epochs >= minimum:
                self.full_train_eligible = True
        elif self.state_epochs >= minimum and ready:
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
            "pu_enabled": self.pu_enabled,
            "pu_disable_reason": self.pu_disable_reason,
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
