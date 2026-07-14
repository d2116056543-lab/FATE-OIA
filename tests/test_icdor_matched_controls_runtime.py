from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from fate_oia.engine.mosaic_icdor_audit_collectors import collect_edge_intervention_audit
from fate_oia.engine.mosaic_target_transfer_metrics import collect_joint_target_transfer_metrics


class _RuntimeControlModel(nn.Module):
    """Small audit model that exposes the actual mask supplied to each forward."""

    def __init__(self) -> None:
        super().__init__()
        self.ontology = {"factors": ({"name": "signal", "type": "point", "spatial": "upper_front"},)}
        self.action_router = SimpleNamespace(
            candidate_edge_mask=torch.ones(2, 1, 1, dtype=torch.bool),
            edge_admission_mask=torch.zeros(2, 1, 1, dtype=torch.bool),
        )
        self.control_overrides: list[torch.Tensor] = []

    def set_edge_admission(self, mask: torch.Tensor) -> None:
        self.action_router.edge_admission_mask = mask.detach().clone()

    def forward(
        self,
        images: torch.Tensor,
        *,
        factor_mask_override: torch.Tensor | None = None,
        factor_intervention_keep_mask: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        assert "action" not in kwargs and "reason" not in kwargs
        masks = torch.zeros(images.shape[0], 1, 10, 10, device=images.device)
        masks[:, 0, 1, 4] = 1.0
        active = masks if factor_mask_override is None else factor_mask_override
        if factor_mask_override is not None:
            self.control_overrides.append(factor_mask_override.detach().cpu())
        if factor_intervention_keep_mask is not None:
            active = active * factor_intervention_keep_mask[:, :, None, None]
        mass = active.sum((-2, -1))[:, 0]
        edge_gain = self.action_router.edge_admission_mask[0, 0, 0].float()
        logits = (mass * (1.0 + edge_gain)).unsqueeze(1)
        return {
            "factor_presence_prob": torch.full((images.shape[0], 1), 0.8, device=images.device),
            "factor_soft_masks": masks,
            "action_final_logits": logits,
            "reason_observed_logits": logits,
        }


def _batch() -> dict[str, object]:
    return {
        "split": ["train_audit"] * 4,
        "image": torch.zeros(4, 3, 10, 10),
        "action": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
        "reason": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
    }


def _assert_real_arms(model: _RuntimeControlModel, metadata: list[dict[str, object]]) -> None:
    assert len(model.control_overrides) >= 4
    arms = model.control_overrides[:4]
    selected = torch.zeros(10, 10, dtype=torch.bool)
    selected[1, 4] = True
    for arm in arms:
        mask = arm[0, 0]
        assert torch.count_nonzero(mask.bool() & selected) == 0
        assert torch.isclose(mask.sum(), torch.tensor(1.0), atol=0.05)
    for left, right in zip(arms, arms[1:]):
        assert torch.count_nonzero(left[0, 0].bool() & right[0, 0].bool()) == 0
    assert len(metadata) == 4
    for arm in metadata:
        assert arm["factor"] == "signal"
        assert arm["factor_type"] == "point"
        assert arm["region"] == "upper_front"
        assert float(arm["max_mass_error"]) <= 0.05
        assert float(arm["max_overlap"]) == 0.0


def test_edge_and_transfer_execute_four_real_same_factor_controls() -> None:
    edge_model = _RuntimeControlModel()
    edge = collect_edge_intervention_audit(
        edge_model,
        [_batch()],
        factor_names=("signal",),
        action_names=("brake",),
        edge_specs=({"factor": "signal", "action": "brake", "direction": "support", "polarity": "present"},),
        device=torch.device("cpu"),
        bootstrap_replicates=8,
    )["edge_stats"]["support:signal->brake"]
    _assert_real_arms(edge_model, edge["matched_control_arms"])

    transfer_model = _RuntimeControlModel()
    transfer = collect_joint_target_transfer_metrics(
        transfer_model,
        [_batch()],
        factor_ids=("signal",),
        action_ids=("brake",),
        reason_ids=("yield",),
        action_directions=(("support",),),
        reason_directions=(("support",),),
        device=torch.device("cpu"),
        route_mode="admitted",
        latent_enabled=True,
    )
    _assert_real_arms(transfer_model, transfer["per_target"][0]["matched_control_arms"])
