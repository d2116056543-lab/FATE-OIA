from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn

from fate_oia.engine.mosaic_icdor_audit_collectors import _edge_metrics, collect_edge_intervention_audit
from fate_oia.engine.mosaic_target_transfer_metrics import collect_joint_target_transfer_metrics


class _RuntimeControlModel(nn.Module):
    """Small audit model that exposes the actual mask supplied to each forward."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
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
        self.forward_calls += 1
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
    arms: list[torch.Tensor] = []
    for packed in model.control_overrides:
        assert packed.shape[0] % 4 == 0
        arms.extend(packed.split(4, dim=0))
    assert len(arms) >= 4
    arms = arms[:4]
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


def test_edge_and_transfer_abstain_without_same_type_identity_control() -> None:
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
    assert all(arm["available_sample_count"] == 0 for arm in edge["matched_control_arms"])
    assert all(arm["control_type"] == "unavailable_noop" for arm in edge["matched_control_arms"])

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
    assert transfer["summary"]["available_pair_count"] == 0
    assert all(
        arm["available_sample_count"] == 0
        for arm in transfer["per_target"][0]["matched_control_arms"]
    )


def test_transfer_abstains_when_dense_mask_cannot_form_four_controls() -> None:
    class DenseControlModel(_RuntimeControlModel):
        def forward(self, images: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            output = super().forward(images, **kwargs)
            if kwargs.get("factor_mask_override") is None:
                output["factor_soft_masks"] = torch.ones(images.shape[0], 1, 10, 10, device=images.device)
            return output

    model = DenseControlModel()
    transfer = collect_joint_target_transfer_metrics(
        model,
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

    assert transfer["summary"]["available_pair_count"] == 0
    assert transfer["per_target"][0]["available"] is False
    assert transfer["per_target"][0]["unavailable_reason"] == "insufficient_matched_control_rows"
    assert transfer["per_target"][0]["tes"] is None


def test_joint_transfer_batches_factor_and_control_interventions() -> None:
    class TwoFactorControlModel(_RuntimeControlModel):
        def __init__(self) -> None:
            super().__init__()
            self.ontology = {
                "factors": (
                    {"name": "signal", "type": "point", "spatial": "upper_front"},
                    {"name": "vehicle", "type": "point", "spatial": "upper_front"},
                )
            }

        def forward(
            self,
            images: torch.Tensor,
            *,
            factor_mask_override: torch.Tensor | None = None,
            factor_intervention_keep_mask: torch.Tensor | None = None,
            **kwargs: object,
        ) -> dict[str, torch.Tensor]:
            self.forward_calls += 1
            masks = torch.zeros(images.shape[0], 2, 10, 10, device=images.device)
            masks[:, 0, 1, 4] = 1.0
            masks[:, 1, 2, 5] = 1.0
            active = masks if factor_mask_override is None else factor_mask_override
            if factor_mask_override is not None:
                self.control_overrides.append(factor_mask_override.detach().cpu())
            if factor_intervention_keep_mask is not None:
                active = active * factor_intervention_keep_mask[:, :, None, None]
            y, x = torch.meshgrid(
                torch.arange(10, device=images.device),
                torch.arange(10, device=images.device),
                indexing="ij",
            )
            spatial_weight = 1.0 + y.float() * 0.1 + x.float() * 0.01
            factor_weight = torch.tensor((1.0, 3.0), device=images.device)
            mass = (active * spatial_weight).sum((-2, -1)).mul(factor_weight).sum(1)
            logits = mass.unsqueeze(1)
            return {
                "factor_presence_prob": torch.full((images.shape[0], 2), 0.8, device=images.device),
                "factor_soft_masks": masks,
                "action_final_logits": logits,
                "reason_observed_logits": logits,
            }

    model = TwoFactorControlModel()
    transfer = collect_joint_target_transfer_metrics(
        model,
        [_batch()],
        factor_ids=("signal", "vehicle"),
        action_ids=("brake",),
        reason_ids=("yield",),
        action_directions=(("support",), ("support",)),
        reason_directions=(("support",), ("support",)),
        device=torch.device("cpu"),
        route_mode="admitted",
        latent_enabled=True,
        intervention_chunk_size=4,
    )

    assert transfer["summary"]["available_pair_count"] == 4
    assert model.forward_calls == 4
    assert transfer["collection_runtime"]["intervention_forward_calls"] == 3
    assert transfer["collection_runtime"]["sequential_intervention_forward_calls"] == 10
    assert transfer["collection_runtime"]["intervention_chunk_size"] == 4
    arm_types = [arm["control_type"] for arm in transfer["per_target"][0]["matched_control_arms"]]
    assert arm_types == ["same_type_identity", "spatial_roll", "spatial_roll", "spatial_roll"]

    sequential_model = TwoFactorControlModel()
    sequential = collect_joint_target_transfer_metrics(
        sequential_model,
        [_batch()],
        factor_ids=("signal", "vehicle"),
        action_ids=("brake",),
        reason_ids=("yield",),
        action_directions=(("support",), ("support",)),
        reason_directions=(("support",), ("support",)),
        device=torch.device("cpu"),
        route_mode="admitted",
        latent_enabled=True,
        intervention_chunk_size=1,
    )
    assert sequential_model.forward_calls == 11
    assert sequential["per_target"] == transfer["per_target"]
    assert sequential["summary"] == transfer["summary"]


def test_training_config_wires_target_transfer_chunk_size() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "configs" / "fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["runtime"]["target_transfer_intervention_chunk_size"] == 4
    source = (root / "fate_oia" / "engine" / "train_acpr_mosaic_trust_icdor.py").read_text(
        encoding="utf-8"
    )
    assert 'intervention_chunk_size=int(config["runtime"]["target_transfer_intervention_chunk_size"])' in source
    assert 'action_ids = {f"action:{name}" for name in model.ontology["action_names"]}' in source


def test_joint_transfer_reuses_and_repeats_batch_local_dino_field() -> None:
    class DinoRuntimeModel(_RuntimeControlModel):
        def __init__(self) -> None:
            super().__init__()
            self.dino_calls = 0

        def dino(self, images: torch.Tensor) -> dict[str, object]:
            self.dino_calls += 1
            return {
                "patch_tokens_by_layer": torch.arange(images.shape[0], dtype=torch.float32)[:, None, None, None],
                "grid_hw": (10, 10),
                "original_tokens": 101,
            }

        def forward(self, images: torch.Tensor, **kwargs: object) -> dict[str, torch.Tensor]:
            field = kwargs.get("precomputed_dino_field")
            assert isinstance(field, dict)
            tokens = field["patch_tokens_by_layer"]
            assert isinstance(tokens, torch.Tensor)
            assert tokens.shape[0] == images.shape[0]
            return super().forward(images, **kwargs)

    model = DinoRuntimeModel()
    collect_joint_target_transfer_metrics(
        model,
        [_batch()],
        factor_ids=("signal",),
        action_ids=("brake",),
        reason_ids=("yield",),
        action_directions=(("support",),),
        reason_directions=(("support",),),
        device=torch.device("cpu"),
        route_mode="admitted",
        latent_enabled=True,
        intervention_chunk_size=4,
    )
    assert model.dino_calls == 1


def test_edge_metrics_require_identity_and_spatial_controls_separately() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    on = torch.tensor([1.0, -1.0, 1.0, -1.0])
    off = torch.tensor([0.0, -1.0, 0.0, -1.0])
    identity = torch.tensor([1.5, -1.0, 1.5, -1.0])
    spatial = torch.tensor([-1.0, -1.0, -1.0, -1.0])

    metrics = _edge_metrics(
        on,
        off,
        (identity + 3.0 * spatial) / 4.0,
        labels,
        direction="support",
        random_identity=identity,
        random_spatial=spatial,
    )

    assert metrics["tes"] > 0.0
    assert metrics["tes_identity"] < 0.0
    assert metrics["tes_spatial"] > 0.0
