"""P17 behavior contract for the sole RAEL optimizer/loss lifecycle owner.

This file is deliberately written before ``train_acpr_rael_oia.py`` exists.
The helpers turn a missing production module into an ordinary pytest failure,
so the initial RED evidence is collectable rather than a collection error.
"""

from __future__ import annotations

import copy
import inspect
import importlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor, nn


ACTION_COUNT = 4
REASON_COUNT = 21
DIM = 8
SLOT_COUNT = 20


def _module():
    try:
        return importlib.import_module("fate_oia.engine.train_acpr_rael_oia")
    except ModuleNotFoundError as error:
        pytest.fail(f"P17 RED: trainer lifecycle module is absent: {error}")


class _TinyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)


class _TinyPUHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(DIM, 1)

    def forward(self, private_delta: Tensor, keep_mask: Tensor) -> Tensor:
        return self.projection(private_delta * keep_mask).squeeze(-1)


class _TinyBridge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # ``internal_proj`` deliberately contains "proj"; P17 must still
        # classify it as internal by module contract rather than name hints.
        self.internal_proj = nn.Linear(DIM, ACTION_COUNT)
        self.gamma_as_raw = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: Tensor) -> Tensor:
        return 0.25 * torch.tanh(self.gamma_as_raw) * self.internal_proj(hidden)


class _TinyUnary(nn.Module):
    def __init__(self, targets: int) -> None:
        super().__init__()
        self.internal_proj = nn.Linear(DIM, targets)
        self.gamma_unary_raw = nn.Parameter(torch.zeros(targets))

    def forward(self, hidden: Tensor) -> Tensor:
        raw = self.internal_proj(hidden)
        return raw * (0.25 * torch.tanh(self.gamma_unary_raw)).view(1, -1)


class _TinyPairwise(nn.Module):
    def __init__(self, targets: int) -> None:
        super().__init__()
        self.targets = targets
        self.internal_proj = nn.Linear(DIM, DIM, bias=False)
        self.pair_output = nn.Parameter(torch.zeros(targets, DIM))
        self.gamma_pair_raw = nn.Parameter(torch.zeros(targets))

    def _raw(self, hidden: Tensor) -> Tensor:
        return torch.nn.functional.linear(torch.tanh(self.internal_proj(hidden)), self.pair_output)

    def forward(self, hidden: Tensor | None = None, **inputs: Tensor) -> dict[str, Tensor]:
        if hidden is None:
            target_summary = inputs["target_tokens"].mean(dim=1)
            evidence_summary = inputs["evidence_tokens"].mean(dim=1)
            hidden = target_summary + evidence_summary
        raw = self._raw(hidden)
        post = raw * (0.25 * torch.tanh(self.gamma_pair_raw)).view(1, -1)
        return {
            "pair_contributions_raw": raw.unsqueeze(-1),
            "pair_contributions": post.unsqueeze(-1),
            "pair_raw_sum": raw,
            "pair_postgamma_sum": post,
        }

    def owner_isolated_auxiliary(self, *, global_logits: Tensor, **inputs: Tensor) -> dict[str, Tensor]:
        isolated = self(**{name: value.detach() for name, value in inputs.items()})
        return {
            "owner_pair_auxiliary_delta": isolated["pair_raw_sum"],
            "owner_auxiliary_logits": global_logits.detach() + isolated["pair_raw_sum"],
        }


class _TinyRAEL(nn.Module):
    """Small real-autograd stand-in with the exact P17 owner names."""

    def __init__(self) -> None:
        super().__init__()
        self.dino_extractor = _TinyDino()
        self.multilayer_field = nn.Linear(DIM, DIM, bias=False)
        self.slot_ledger = nn.Linear(DIM, DIM, bias=False)
        self.slot_attribute_heads = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM))
        self.action_category = nn.Linear(DIM, ACTION_COUNT)
        self.semantic_reason = nn.Embedding(REASON_COUNT, DIM)
        self.action_reason_bridge = _TinyBridge()
        self.action_unary = _TinyUnary(ACTION_COUNT)
        self.reason_unary = _TinyUnary(REASON_COUNT)
        self.action_pairwise = _TinyPairwise(ACTION_COUNT)
        self.reason_pairwise = _TinyPairwise(REASON_COUNT)
        self.reason_private = nn.Linear(DIM, REASON_COUNT)
        self.pu_private_head = _TinyPUHead()
        self.calibration = nn.Parameter(torch.zeros(REASON_COUNT))
        self.p17_pairwise_replay = True
        self.register_buffer("_pu_active_labels", torch.zeros(REASON_COUNT, dtype=torch.bool))
        self.register_buffer("_pu_feature_keep_view_one", torch.ones(1, 1, DIM))
        self.register_buffer("_pu_feature_keep_view_two", torch.full((1, 1, DIM), 0.9))

    def set_pu_active_labels(self, active: Tensor) -> None:
        self._pu_active_labels.copy_(active)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        hidden = self.multilayer_field(images)
        # Keep all public ranges as views of one canonical [B,20,D] boundary,
        # matching P16's explicit ``outputs['evidence_slots']`` contract.
        evidence_slots = self.slot_ledger(hidden).unsqueeze(1).repeat(1, SLOT_COUNT, 1)
        slots = evidence_slots
        semantic = self.semantic_reason.weight.unsqueeze(0).expand(images.shape[0], -1, -1)
        slot_summary = evidence_slots.mean(dim=1)
        semantic = semantic + hidden.unsqueeze(1) + 0.1 * slot_summary.unsqueeze(1)
        # The formal P17 path must consume both shared admission boundaries;
        # otherwise a registered evidence-slot hook correctly never fires.
        action_global = self.action_category(hidden + 0.1 * slot_summary)
        # The action path may read public semantic evidence, but never the
        # private residual.  This mirrors the P16 action firewall.
        action_pair = self.action_pairwise(hidden)
        reason_pair = self.reason_pairwise(hidden)
        action_final = (
            action_global
            + self.action_reason_bridge(hidden + semantic.mean(dim=1))
            + self.action_unary(hidden)
            + action_pair["pair_postgamma_sum"]
        )
        reason_global = self.reason_private(hidden)
        reason_final = reason_global + self.reason_unary(hidden) + reason_pair["pair_postgamma_sum"]
        private_delta = hidden.unsqueeze(1).expand(-1, REASON_COUNT, -1)
        attribute_hidden = self.slot_attribute_heads(hidden)
        entity_presence = attribute_hidden.mean(dim=-1, keepdim=True).expand(-1, 12)
        entity_type = attribute_hidden[:, :6].unsqueeze(1).expand(-1, 12, -1)
        traffic_state = attribute_hidden[:, :4].unsqueeze(1).expand(-1, 12, -1)
        entity_reliability = torch.sigmoid(entity_presence)
        road_bias = attribute_hidden.mean(dim=-1).view(-1, 1, 1, 1)
        drivable_logits = slots[:, 12:15].mean(dim=-1).view(-1, 3, 1, 1) + road_bias
        boundary_logits = slots[:, 15:17].mean(dim=-1).view(-1, 2, 1, 1) + road_bias
        boundary_style_logits = attribute_hidden[:, :3].unsqueeze(1).expand(-1, 2, -1)
        latent_slots = slots[:, 17:20]
        first_keep = torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            device=hidden.device,
        ).view(1, 1, DIM)
        second_keep = torch.tensor(
            [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            device=hidden.device,
        ).view(1, 1, DIM)
        private_logits = self.pu_private_head(private_delta, self._pu_feature_keep_view_one)
        private_probabilities = torch.sigmoid(private_logits)
        return {
            "action_logits_global": action_global,
            "action_logits_final": action_final,
            "reason_logits_global": reason_global,
            "reason_logits_final": reason_final,
            "semantic_reason_tokens": semantic,
            "evidence_slots": evidence_slots,
            "entity_slots": slots[:, :12],
            "road_slots": slots[:, 12:17],
            "latent_slots": slots[:, 17:20],
            "latent_feature_view_one": latent_slots * first_keep,
            "latent_feature_view_two": latent_slots * second_keep,
            "grounding_outputs": {
                "entity": {
                    "presence_logits": entity_presence,
                    "entity_type_logits": entity_type,
                    "traffic_state_logits": traffic_state,
                    "entity_reliability": entity_reliability,
                },
                "road": {
                    "drivable_logits": drivable_logits,
                    "boundary_logits": boundary_logits,
                    "boundary_style_logits": boundary_style_logits,
                    "drivable_reliability": torch.ones(
                        images.shape[0], 3, device=hidden.device
                    ),
                    "boundary_reliability": torch.ones(
                        images.shape[0], 2, device=hidden.device
                    ),
                },
            },
            "reason_private_delta": private_delta,
            "pu_scores": private_probabilities.detach(),
            "pu_active_labels": self._pu_active_labels,
            "action_pair_delta": action_pair["pair_postgamma_sum"],
            "reason_pair_delta": reason_pair["pair_postgamma_sum"],
            "action_pair_delta_raw": action_pair["pair_raw_sum"],
            "reason_pair_delta_raw": reason_pair["pair_raw_sum"],
            "action_tokens": hidden.unsqueeze(1).expand(-1, ACTION_COUNT, -1),
            "action_slot_weights": torch.full(
                (images.shape[0], ACTION_COUNT, 21),
                1.0 / 21.0,
                device=hidden.device,
            ),
            "reason_slot_weights": torch.full(
                (images.shape[0], REASON_COUNT, 21),
                1.0 / 21.0,
                device=hidden.device,
            ),
            "slot_reliability": torch.ones(
                images.shape[0], SLOT_COUNT, device=hidden.device
            ),
            "slot_observability": torch.ones(
                images.shape[0], SLOT_COUNT, device=hidden.device
            ),
            "slot_masks": torch.ones(
                images.shape[0], SLOT_COUNT, 1, 1, device=hidden.device
            ),
            "slot_centroid": torch.zeros(
                images.shape[0], SLOT_COUNT, 2, device=hidden.device
            ),
            "slot_scale": torch.full(
                (images.shape[0], SLOT_COUNT), 0.2, device=hidden.device
            ),
            "slot_type_probs": torch.nn.functional.one_hot(
                torch.zeros(
                    images.shape[0], 12, dtype=torch.long, device=hidden.device
                ),
                num_classes=6,
            ).float(),
            "slot_state_probs": torch.nn.functional.one_hot(
                torch.zeros(
                    images.shape[0], 12, dtype=torch.long, device=hidden.device
                ),
                num_classes=4,
            ).float(),
            "slot_sector_probs": {
                "horizontal": torch.ones(
                    images.shape[0], SLOT_COUNT, 3, device=hidden.device
                )
                / 3.0
            },
            "reason_unary_contributions_raw": torch.zeros(
                images.shape[0], REASON_COUNT, SLOT_COUNT, device=hidden.device
            ),
            "diagnostics": {
                "pu": {
                    "private_logits_view_one": private_logits,
                    "private_logits_view_two": private_logits + 0.01,
                    "p_evidence": private_probabilities.detach(),
                    "p_private": private_probabilities.detach(),
                    "p_private_view_one": private_probabilities.detach(),
                    "p_private_view_two": private_probabilities.detach(),
                    "c_view": torch.ones_like(private_probabilities),
                    "c_obs": torch.ones_like(private_probabilities),
                }
            },
        }

    def encode_images(self, images: Tensor) -> Tensor:
        return images

    def decode_from_field_provisional(self, field: Tensor) -> dict[str, Any]:
        return self.forward(field)

    def decode_from_field_with_reliability(
        self,
        field: Tensor,
        *,
        q_ground: Tensor,
        q_view: Tensor,
        q_view_sector: Tensor,
    ) -> dict[str, Any]:
        output = self.forward(field)
        q_state = torch.ones_like(q_ground)
        rho = (
            output["slot_observability"].detach()
            * q_ground.detach()
            * q_view.detach()
            * q_state
        )
        output.update(
            {
                "slot_q_ground": q_ground.detach(),
                "slot_q_view": q_view.detach(),
                "slot_q_state": q_state,
                "slot_reliability": rho,
                "road_rho_clear": q_view_sector.detach(),
            }
        )
        return output


class _RNGTinyRAEL(_TinyRAEL):
    """Consumes every checkpointed RNG stream without changing the contract."""

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        random.random()
        np.random.rand()
        torch.rand(())
        if torch.cuda.is_available():
            torch.rand((), device="cuda")
        return super().forward(images)


class _TransactionalTinyRAEL(_RNGTinyRAEL):
    """Adds forward-side mutable state that a failed boundary must roll back."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.register_buffer("forward_side_counter", torch.zeros((), dtype=torch.long))

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        self.forward_calls += 1
        self.forward_side_counter.add_(1)
        return super().forward(images)


class _CounterfactualTinyRAEL(_TinyRAEL):
    def __init__(self) -> None:
        super().__init__()
        self.encode_calls = 0
        self.decode_calls = 0
        self.replay_build_calls = 0

    def encode_images(self, images: Tensor) -> dict[str, Tensor]:
        self.encode_calls += 1
        spatial_bias = torch.linspace(
            -0.5, 0.5, 20, device=images.device, dtype=images.dtype
        ).view(1, 1, 1, 20)
        shared = images.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 20)
        shared = shared + spatial_bias - spatial_bias.mean()
        return {"images": images, "shared_field": shared}

    def decode_from_field(self, field: dict[str, Tensor]) -> dict[str, Any]:
        self.decode_calls += 1
        outputs = super().forward(field["images"])
        batch = field["images"].shape[0]
        masks = torch.zeros(
            batch, SLOT_COUNT, 4, 20, device=field["images"].device
        )
        for slot in range(SLOT_COUNT):
            masks[:, slot, :, slot] = 1.0
        outputs["slot_masks"] = masks
        outputs["slot_sector_probs"] = {
            "horizontal": torch.nn.functional.one_hot(
                torch.ones(
                    batch,
                    SLOT_COUNT,
                    dtype=torch.long,
                    device=field["images"].device,
                ),
                num_classes=3,
            ).float()
        }
        action_deletion = torch.zeros(
            batch, ACTION_COUNT, SLOT_COUNT, device=field["images"].device
        )
        reason_deletion = torch.zeros(
            batch, REASON_COUNT, SLOT_COUNT, device=field["images"].device
        )
        action_deletion[:, 0, 0] = 1.0
        reason_deletion[:, 0, 0] = 1.0
        outputs["action_analytical_deletion"] = action_deletion
        outputs["reason_analytical_deletion"] = reason_deletion
        outputs["_cf_shared_field"] = field["shared_field"]
        return outputs

    def build_counterfactual_replay(
        self,
        field: dict[str, Tensor],
        outputs: dict[str, Any],
        *,
        target_family: str,
    ) -> dict[str, Any]:
        self.replay_build_calls += 1
        shared = field["shared_field"]
        base = outputs[f"{target_family}_logits_final"]
        baseline_mean = shared.mean(dim=(-1, -2))
        target_count = base.shape[1]

        def readout(replay_field: Tensor) -> Tensor:
            delta = (
                replay_field.mean(dim=(-1, -2)) - baseline_mean
            )[:, :target_count]
            return base + 0.1 * delta

        return {
            "shared_field": shared,
            "public_readout": readout,
            "public_contribution": lambda _field: torch.zeros_like(base),
        }


def _batch(batch_size: int = 3) -> dict[str, Tensor]:
    torch.manual_seed(7)
    return {
        "images": torch.randn(batch_size, DIM),
        "action_targets": (torch.rand(batch_size, ACTION_COUNT) > 0.5).float(),
        "reason_targets": (torch.rand(batch_size, REASON_COUNT) > 0.7).float(),
        "grounding_mode": "synthetic_prebuilt",
        "grounding_targets": {
            "entity": {
                "presence": torch.ones(batch_size, 12),
                "presence_valid": torch.ones(batch_size, 12, dtype=torch.bool),
                "type": torch.zeros(batch_size, 12, dtype=torch.long),
                "type_valid": torch.ones(batch_size, 12, dtype=torch.bool),
                "traffic_state": torch.full(
                    (batch_size, 12), -1, dtype=torch.long
                ),
                "traffic_state_valid": torch.zeros(
                    batch_size, 12, dtype=torch.bool
                ),
            },
            "road": {
                "drivable_targets": torch.ones(batch_size, 3, 1, 1),
                "drivable_valid_mask": torch.ones(
                    batch_size, 3, dtype=torch.bool
                ),
                "boundary_targets": torch.ones(batch_size, 2, 1, 1),
                "boundary_valid_mask": torch.ones(
                    batch_size, 2, dtype=torch.bool
                ),
                "boundary_style_targets": torch.zeros(
                    batch_size, 2, dtype=torch.long
                ),
                "boundary_style_valid_mask": torch.ones(
                    batch_size, 2, dtype=torch.bool
                ),
            },
        },
        "mirror_pairs": (
            torch.tensor([[0, 1]], dtype=torch.long)
            if batch_size > 1
            else torch.empty(0, 2, dtype=torch.long)
        ),
        "file_names": tuple(
            f"E:\\data\\rael_case_{index:04d}.jpg"
            for index in range(batch_size)
        ),
    }


def _dynamic_batch(batch_size: int = 1) -> dict[str, Any]:
    from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord

    batch = _batch(batch_size)
    batch.pop("grounding_targets")
    batch["grounding_mode"] = "dynamic"
    batch["grounding_records"] = tuple(
        RAELGroundingRecord(
            detections=(
                {
                    "id": "shared-object",
                    "category": "vehicle",
                    "box": (16.0, 8.0, 40.0, 28.0),
                    "sector": "left",
                    "attributes": {},
                },
            ),
            lanes=(),
            drivable=(),
            source_complete={
                "detections": True,
                "lanes": False,
                "drivable": False,
            },
        )
        for _ in range(batch_size)
    )
    batch["transform_meta"] = tuple(
        {"image_size": (64, 32), "mirror": False, "object_ids": (f"obj-{index}",)}
        for index in range(batch_size)
    )
    return batch


def _independent_grounding_values(
    module: Any,
    outputs: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[Tensor, Tensor]:
    from fate_oia.losses.rael_grounding_losses import (
        entity_attribute_grounding_loss,
        road_grounding_loss_bundle,
    )

    targets = batch["grounding_targets"]
    entity_results = entity_attribute_grounding_loss(
        outputs["grounding_outputs"]["entity"], targets["entity"]
    )
    road_outputs = outputs["grounding_outputs"]["road"]
    road_targets = targets["road"]
    road_results = road_grounding_loss_bundle(
        drivable_logits=road_outputs["drivable_logits"],
        drivable_targets=road_targets["drivable_targets"],
        drivable_valid_mask=road_targets["drivable_valid_mask"],
        boundary_logits=road_outputs["boundary_logits"],
        boundary_targets=road_targets["boundary_targets"],
        boundary_valid_mask=road_targets["boundary_valid_mask"],
        boundary_style_logits=road_outputs["boundary_style_logits"],
        boundary_style_targets=road_targets["boundary_style_targets"],
        boundary_style_valid_mask=road_targets["boundary_style_valid_mask"],
        drivable_reliability=road_outputs["drivable_reliability"],
        boundary_reliability=road_outputs["boundary_reliability"],
    )
    entity = sum(
        (result.loss for result in entity_results.values()),
        outputs["action_logits_final"].sum() * 0.0,
    )
    road = sum(
        (result.loss for result in road_results.values()),
        outputs["action_logits_final"].sum() * 0.0,
    )

    masks = outputs["slot_masks"]
    pairs = batch["mirror_pairs"].to(masks.device)
    if pairs.numel():
        permutation = torch.tensor(
            [*range(12), 14, 13, 12, 16, 15, 17, 18, 19],
            device=masks.device,
        )
        canonical = masks.index_select(0, pairs[:, 0])
        mirrored = (
            masks.index_select(0, pairs[:, 1])
            .index_select(1, permutation)
            .flip(-1)
        )
        road_view = (canonical[:, 12:17] - mirrored[:, 12:17]).abs().mean()
        entity_view = (
            canonical[:, :12].sum(1) - mirrored[:, :12].sum(1)
        ).abs().mean()
        latent_view = (
            canonical[:, 17:20].sum(1) - mirrored[:, 17:20].sum(1)
        ).abs().mean()
        ground_view = (road_view + entity_view + latent_view) / 3.0
    else:
        ground_view = masks.sum() * 0.0

    latent = torch.nn.functional.normalize(
        outputs["latent_slots"].float(), dim=-1
    )
    similarity = torch.einsum("bjd,bkd->bjk", latent, latent)
    off_diagonal = ~torch.eye(3, dtype=torch.bool, device=latent.device)
    slot_diversity = similarity[:, off_diagonal].square().mean()
    feature_view = (
        1.0
        - torch.nn.functional.cosine_similarity(
            outputs["latent_feature_view_one"].float(),
            outputs["latent_feature_view_two"].float(),
            dim=-1,
        )
    ).mean()
    grounding = entity + road + 0.10 * ground_view + 0.02 * slot_diversity
    return grounding, feature_view


def _named_parameter_snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _assert_nested_exact(left: Any, right: Any, path: str = "state") -> None:
    if isinstance(left, Tensor):
        assert isinstance(right, Tensor), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray), path
        assert np.array_equal(left, right), path
        return
    if isinstance(left, dict):
        assert isinstance(right, dict) and set(left) == set(right), path
        for key in left:
            _assert_nested_exact(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_nested_exact(left_item, right_item, f"{path}[{index}]")
        return
    assert left == right, path


def test_p17_behavior_red_collects_missing_trainer_as_an_ordinary_failure() -> None:
    """The pre-implementation run must fail here, not during collection."""
    module = _module()
    assert hasattr(module, "RAELTrainer")


def test_p17_single_warmup_weights_exactly_follow_r5_and_r10() -> None:
    module = _module()
    schedule = module.RAELWarmupSchedule(total_optimizer_updates=100)

    initial = schedule.weights(0)
    at_r5 = schedule.weights(5)
    at_r10 = schedule.weights(10)
    assert initial.r5 == pytest.approx(0.0)
    assert initial.r10 == pytest.approx(0.0)
    assert initial.grounding == pytest.approx(0.05)
    assert initial.non_regression == pytest.approx(0.02)
    assert initial.pairwise_auxiliary == pytest.approx(0.0)
    assert initial.counterfactual == pytest.approx(0.0)
    assert initial.feature_view == pytest.approx(0.0)
    assert at_r5.r5 == pytest.approx(1.0)
    assert at_r5.r10 == pytest.approx(0.5)
    assert at_r5.grounding == pytest.approx(0.15)
    assert at_r5.non_regression == pytest.approx(0.05)
    assert at_r5.feature_view == pytest.approx(0.02)
    assert at_r10.r10 == pytest.approx(1.0)
    assert at_r10.pairwise_auxiliary == pytest.approx(0.05)
    assert at_r10.counterfactual == pytest.approx(0.05)


def test_p17_owner_groups_are_exact_unique_and_calibration_is_excluded() -> None:
    module = _module()
    model = _TinyRAEL()
    bundle = module.build_rael_optimizer(model)
    expected = {
        "multilayer_field",
        "slot_ledger_core",
        "slot_attribute_heads",
        "action_category",
        "semantic_reason",
        "action_reason_bridge",
        "unary_contribution",
        "pairwise_relation",
        "reason_private",
        "pu_private",
    }
    assert set(bundle.owner_parameter_names) == expected
    assert bundle.owner_learning_rates["reason_private"] == pytest.approx(3.0e-4)
    assert bundle.owner_learning_rates["pu_private"] == pytest.approx(3.0e-4)
    assert all(bundle.owner_learning_rates[name] == pytest.approx(2.0e-4) for name in expected - {"reason_private", "pu_private"})
    seen = [name for names in bundle.owner_parameter_names.values() for name in names]
    assert len(seen) == len(set(seen))
    assert "calibration" not in seen
    assert all("dino_extractor" not in name for name in seen)
    no_decay = set(bundle.no_decay_parameter_names)
    assert "slot_attribute_heads.0.weight" in no_decay
    assert "slot_attribute_heads.0.bias" in no_decay
    assert "semantic_reason.weight" in no_decay
    assert "action_category.bias" in no_decay


def test_p17_loss_is_independently_recomputed_as_fourteen_exact_once_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=100, gradient_accumulation_steps=1)
    trainer.set_pu_label_gate(
        torch.ones(REASON_COUNT, dtype=torch.bool),
        torch.full((REASON_COUNT,), 0.2),
    )
    batch = _batch()
    outputs = model(batch["images"])
    originals = {
        name: getattr(module, name)
        for name in (
            "multilabel_asymmetric_loss",
            "evidence_conditional_loss",
            "two_way_consistency_loss",
            "soft_f1_loss",
            "multilabel_ranking_loss",
            "reason_private_pu_loss",
        )
    }
    calls = {name: 0 for name in originals}

    for loss_name, original in originals.items():
        def counted(*args: Any, _name: str = loss_name, _original=original, **kwargs: Any):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(module, loss_name, counted)

    bundle = trainer.compute_loss_bundle(
        outputs,
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        counterfactual_loss=torch.tensor(0.7),
        optimizer_step=10,
        epoch=1,
    )
    assert calls == {
        "multilabel_asymmetric_loss": 4,
        "evidence_conditional_loss": 2,
        "two_way_consistency_loss": 2,
        "soft_f1_loss": 1,
        "multilabel_ranking_loss": 1,
        "reason_private_pu_loss": 1,
    }

    action_targets = batch["action_targets"]
    reason_targets = batch["reason_targets"]
    pu = outputs["diagnostics"]["pu"]
    soft = module.build_pu_soft_targets(
        reason_targets,
        pu["p_evidence"],
        pu["p_private"],
        pu["c_view"],
        pu["c_obs"],
        torch.full((REASON_COUNT,), 0.2),
        update_index=trainer.optimizer_step,
    )
    confidence = module.reason_confidence_weights(
        outputs["reason_slot_weights"][..., :20],
        outputs["slot_reliability"],
        outputs["reason_unary_contributions_raw"],
        pu["p_private_view_one"],
        pu["p_private_view_two"],
        outputs["slot_observability"].mean(dim=1, keepdim=True).expand_as(reason_targets),
    )
    private_delta = outputs["reason_private_delta"].detach()
    private_logits = 0.5 * (
        model.pu_private_head(private_delta, model._pu_feature_keep_view_one)
        + model.pu_private_head(private_delta, model._pu_feature_keep_view_two)
    )
    action_owner = model.action_pairwise.owner_isolated_auxiliary(
        global_logits=outputs["action_logits_global"],
        target_tokens=outputs["action_tokens"],
        evidence_tokens=outputs["evidence_slots"],
        slot_masks=outputs["slot_masks"],
        sector_probs=outputs["slot_sector_probs"]["horizontal"],
        unary_public_pi=outputs["action_slot_weights"][..., :20],
        reliability=outputs["slot_reliability"],
    )
    reason_owner = model.reason_pairwise.owner_isolated_auxiliary(
        global_logits=outputs["reason_logits_global"],
        target_tokens=outputs["semantic_reason_tokens"] + outputs["reason_private_delta"].detach(),
        evidence_tokens=outputs["evidence_slots"],
        slot_masks=outputs["slot_masks"],
        sector_probs=outputs["slot_sector_probs"]["horizontal"],
        unary_public_pi=outputs["reason_slot_weights"][..., :20],
        reliability=outputs["slot_reliability"],
    )
    action_final = originals["multilabel_asymmetric_loss"](outputs["action_logits_final"], action_targets)
    action_global = originals["multilabel_asymmetric_loss"](outputs["action_logits_global"], action_targets)
    reason_final = originals["evidence_conditional_loss"](
        outputs["reason_logits_final"], soft["soft_targets"], confidence["positive_weight"], confidence["negative_weight"]
    )
    reason_global = originals["multilabel_asymmetric_loss"](outputs["reason_logits_global"], reason_targets)
    pair_auxiliary = originals["multilabel_asymmetric_loss"](
        action_owner["owner_auxiliary_logits"], action_targets
    ) + originals["evidence_conditional_loss"](
        reason_owner["owner_auxiliary_logits"], reason_targets,
        confidence["positive_weight"], confidence["negative_weight"],
    )
    grounding_value, feature_view_value = _independent_grounding_values(
        module, outputs, batch
    )
    fourteen_terms = {
        "action_final": action_final,
        "action_global_half": 0.5 * action_global,
        "action_consistency": 0.05 * originals["two_way_consistency_loss"](
            outputs["action_logits_final"], outputs["action_logits_global"]
        ),
        "action_soft_f1": 0.05 * originals["soft_f1_loss"](outputs["action_logits_final"], action_targets),
        "reason_final": reason_final,
        "reason_global_half": 0.5 * reason_global,
        "reason_rank": 0.05 * originals["multilabel_ranking_loss"](outputs["reason_logits_final"], reason_targets),
        "reason_consistency": 0.05 * originals["two_way_consistency_loss"](
            outputs["reason_logits_final"], outputs["reason_logits_global"]
        ),
        "grounding": 0.15 * grounding_value,
        "pairwise": 0.05 * pair_auxiliary,
        "counterfactual": 0.05 * torch.tensor(0.7),
        "non_regression": 0.05 * torch.relu(action_final - action_global.detach() + 0.002),
        "feature_view": 0.02 * feature_view_value,
        "pu_private": originals["reason_private_pu_loss"](
            private_logits, soft["soft_targets"], confidence["positive_weight"], confidence["negative_weight"]
        ),
    }
    assert len(fourteen_terms) == 14
    expected = sum(fourteen_terms.values(), torch.zeros(()))
    assert torch.allclose(bundle.total, expected, rtol=0.0, atol=1.0e-7)
    assert bundle.pairwise_auxiliary.requires_grad
    assert torch.allclose(
        bundle.components["pairwise_auxiliary_weighted"],
        0.05 * pair_auxiliary,
    )


def test_p17_active_pu_private_is_present_exactly_once_in_total_loss() -> None:
    """A nonzero PU gate must add one, and only one, private-owner loss."""
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=100, gradient_accumulation_steps=1)
    trainer.set_pu_label_gate(
        torch.ones(REASON_COUNT, dtype=torch.bool),
        torch.full((REASON_COUNT,), 0.2),
    )
    batch = _batch()
    bundle = trainer.compute_loss_bundle(
        model(batch["images"]),
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        counterfactual_loss=torch.tensor(0.7),
        optimizer_step=10,
        epoch=1,
    )
    without_pu = (
        bundle.action
        + bundle.reason
        + bundle.weights.grounding * bundle.grounding
        + bundle.weights.pairwise_auxiliary * bundle.pairwise_auxiliary
        + bundle.weights.counterfactual * bundle.counterfactual
        + bundle.weights.non_regression * bundle.non_regression
        + bundle.weights.feature_view * bundle.feature_view
    )
    assert bundle.pu_private.item() > 0.0
    assert torch.allclose(bundle.total - without_pu, bundle.pu_private)


def test_p17_grounding_is_computed_from_current_outputs_and_static_targets() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(
        model, total_optimizer_updates=100, gradient_accumulation_steps=1
    )
    batch = _batch()
    outputs = model(batch["images"])
    expected_grounding, expected_feature_view = _independent_grounding_values(
        module, outputs, batch
    )
    bundle = trainer.compute_loss_bundle(
        outputs,
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        optimizer_step=5,
        epoch=1,
    )
    torch.testing.assert_close(bundle.grounding, expected_grounding)
    torch.testing.assert_close(bundle.feature_view, expected_feature_view)
    grounding_grads = torch.autograd.grad(
        bundle.grounding,
        (
            model.slot_attribute_heads[1].weight,
            model.slot_ledger.weight,
        ),
        retain_graph=True,
    )
    assert all(torch.count_nonzero(gradient).item() > 0 for gradient in grounding_grads)
    feature_grad = torch.autograd.grad(
        bundle.feature_view, model.slot_ledger.weight
    )[0]
    assert torch.count_nonzero(feature_grad).item() > 0


def test_p17_dynamic_grounding_is_built_post_forward_from_current_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    trainer = module.RAELTrainer(
        _TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    batch = _dynamic_batch()
    original = module.build_dynamic_grounding_batch
    calls: list[tuple[Any, ...]] = []

    def capture(slots: Any, records: Any, image_sizes: Any):
        calls.append((slots, records, image_sizes))
        return original(slots, records, image_sizes)

    monkeypatch.setattr(module, "build_dynamic_grounding_batch", capture)
    result = trainer.train_microbatch(batch, epoch=1)

    assert torch.isfinite(result.components["total"])
    assert len(calls) == 1
    descriptors, records, image_sizes = calls[0]
    assert len(descriptors) == len(records) == len(image_sizes) == 1
    assert len(descriptors[0]) == 12
    assert descriptors[0][0]["category"] == "vehicle"
    assert image_sizes == ((64, 32),)
    assert trainer.last_dynamic_grounding_batch is not None
    assert trainer.last_dynamic_grounding_batch.coverage["matched_entity_count"] == 1


def test_p17_dynamic_grounding_rejects_missing_transformed_records_or_meta() -> None:
    module = _module()
    trainer = module.RAELTrainer(
        _TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    missing_records = _dynamic_batch()
    missing_records.pop("grounding_records")
    with pytest.raises(ValueError, match="grounding_records"):
        trainer.train_microbatch(missing_records, epoch=1)

    missing_meta = _dynamic_batch()
    missing_meta.pop("transform_meta")
    with pytest.raises(ValueError, match="transform_meta"):
        trainer.train_microbatch(missing_meta, epoch=1)


def test_p17_dynamic_reliability_updates_object_aligned_view_ema() -> None:
    module = _module()
    trainer = module.RAELTrainer(
        _TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    trainer.train_microbatch(_dynamic_batch(2), epoch=1)

    result = trainer.last_dynamic_reliability
    assert result is not None
    assert result.q_view[0, 0] > 0.0
    assert result.q_view[1, 0] > 0.0
    assert torch.count_nonzero(result.q_view[:, 17:20]).item() == 0
    assert trainer._view_ema_state["objects"]["shared-object"] > 0.0


def test_p17_prebuilt_grounding_requires_explicit_synthetic_mode() -> None:
    module = _module()
    trainer = module.RAELTrainer(
        _TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    batch = _batch()
    batch.pop("grounding_mode")
    with pytest.raises(ValueError, match="grounding_records"):
        trainer.train_microbatch(batch, epoch=1)


def test_p17_rejects_prebuilt_grounding_term_constants() -> None:
    module = _module()
    trainer = module.RAELTrainer(
        _TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    batch = _batch()
    batch.pop("grounding_targets")
    batch["grounding_terms"] = {
        name: torch.tensor(value)
        for name, value in {
            "entity": 0.2,
            "road": 0.3,
            "view": 0.4,
            "slot_diversity": 0.5,
            "mirror_view": 0.6,
        }.items()
    }
    with pytest.raises(ValueError, match="grounding_targets"):
        trainer.train_microbatch(batch, epoch=1)


def test_p17_epoch_zero_disables_hidden_recovery_but_still_supervises_private_owner() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    batch = _batch()
    epoch_zero = trainer.train_microbatch(batch, epoch=0)
    assert not bool(model._pu_active_labels.any())
    assert epoch_zero.components["pu_private"].item() > 0.0
    assert epoch_zero.owner_gradient_norms_pre_clip["pu_private"] > 0.0


def test_p17_lambda_zero_and_all_off_use_observed_targets_for_private_supervision() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    trainer.set_pu_label_gate(torch.zeros(REASON_COUNT, dtype=torch.bool), torch.zeros(REASON_COUNT))
    batch = _batch()
    outputs = model(batch["images"])
    bundle = trainer.compute_loss_bundle(
        outputs,
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        epoch=1,
    )
    assert bundle.pu_private.item() > 0.0
    private_grad = torch.autograd.grad(bundle.pu_private, tuple(model.pu_private_head.parameters()), allow_unused=False)
    assert sum(gradient.abs().sum().item() for gradient in private_grad) > 0.0


def test_p17_reason_pair_auxiliary_uses_the_same_pu_soft_targets_as_main_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(
        model, total_optimizer_updates=20, gradient_accumulation_steps=1
    )
    trainer.set_pu_label_gate(
        torch.ones(REASON_COUNT, dtype=torch.bool),
        torch.full((REASON_COUNT,), 0.2),
    )
    trainer.optimizer_step = 5
    captured_targets: list[Tensor] = []
    original = module.evidence_conditional_loss

    def capture(
        logits: Tensor,
        targets: Tensor,
        positive_weight: Tensor,
        negative_weight: Tensor,
    ) -> Tensor:
        captured_targets.append(targets.detach().clone())
        return original(logits, targets, positive_weight, negative_weight)

    monkeypatch.setattr(module, "evidence_conditional_loss", capture)
    batch = _batch()
    trainer.compute_loss_bundle(
        model(batch["images"]),
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        optimizer_step=5,
        epoch=1,
    )
    assert len(captured_targets) == 2
    assert not torch.equal(captured_targets[0], batch["reason_targets"])
    torch.testing.assert_close(
        captured_targets[1], captured_targets[0], rtol=0.0, atol=0.0
    )



def test_p17_action_private_firewall_and_reason_action_firewall_hold_at_parameter_level() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    batch = _batch()
    matrix = trainer.loss_owner_gradient_matrix(batch, epoch=1)
    assert matrix["action"]["reason_private"] == pytest.approx(0.0, abs=1e-10)
    assert matrix["action"]["pu_private"] == pytest.approx(0.0, abs=1e-10)
    assert matrix["reason"]["action_category"] == pytest.approx(0.0, abs=1e-10)
    assert matrix["reason"]["action_reason_bridge"] == pytest.approx(0.0, abs=1e-10)


def test_p17_pairwise_uses_only_declared_r10_weight_and_natural_update_two_three_startup() -> None:
    module = _module()
    model = _TinyRAEL()
    assert torch.count_nonzero(model.action_pairwise.pair_output) == 0
    assert torch.count_nonzero(model.reason_pairwise.pair_output) == 0
    assert torch.count_nonzero(model.action_pairwise.gamma_pair_raw) == 0
    assert torch.count_nonzero(model.reason_pairwise.gamma_pair_raw) == 0
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    batch = _batch()
    bundle = trainer.compute_loss_bundle(
        model(batch["images"]),
        action_targets=batch["action_targets"],
        reason_targets=batch["reason_targets"],
        grounding_targets=batch["grounding_targets"],
        mirror_pairs=batch["mirror_pairs"],
        optimizer_step=0,
        epoch=0,
    )
    assert bundle.weights.pairwise_auxiliary == pytest.approx(0.0)
    assert bundle.components["pairwise_auxiliary_weighted"].item() == pytest.approx(0.0)
    pair_parameters = tuple(model.action_pairwise.parameters()) + tuple(
        model.reason_pairwise.parameters()
    )
    step_zero_grads = torch.autograd.grad(
        bundle.total, pair_parameters, allow_unused=True
    )
    assert all(
        gradient is None or torch.count_nonzero(gradient).item() == 0
        for gradient in step_zero_grads
    )

    update_one = trainer.train_microbatch(batch, epoch=0)
    assert update_one.owner_task_gradient_norms_pre_clip["pairwise_relation"] == pytest.approx(0.0)
    assert update_one.owner_optimizer_step_count["pairwise_relation"] == 0
    update_two = trainer.train_microbatch(batch, epoch=0)
    assert update_two.owner_task_gradient_norms_pre_clip["pairwise_relation"] > 0.0
    assert update_two.owner_optimizer_step_count["pairwise_relation"] == 1
    assert torch.count_nonzero(model.action_pairwise.pair_output) + torch.count_nonzero(model.reason_pairwise.pair_output) > 0
    update_three = trainer.train_microbatch(batch, epoch=0)
    assert update_three.owner_optimizer_step_count["pairwise_relation"] == 2
    records = trainer.bootstrap.state_dict()["records"]
    assert records[2]["pairwise_internal"] > 0.0


def test_p17_rezero_partition_is_explicit_and_never_uses_proj_name_heuristics() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, 2.0)
    for parameter in (
        model.action_reason_bridge.gamma_as_raw,
        model.action_unary.gamma_unary_raw,
        model.reason_unary.gamma_unary_raw,
        model.action_pairwise.pair_output,
        model.reason_pairwise.pair_output,
        model.action_pairwise.gamma_pair_raw,
        model.reason_pairwise.gamma_pair_raw,
    ):
        parameter.grad = torch.ones_like(parameter)
    norms = trainer._rezero_norms()
    bridge_output_expected = model.action_reason_bridge.gamma_as_raw.grad.float().norm().item()
    bridge_internal_expected = sum(
        parameter.grad.float().square().sum()
        for name, parameter in model.action_reason_bridge.named_parameters()
        if name != "gamma_as_raw"
    ).sqrt().item()
    assert norms["bridge_output"] == pytest.approx(bridge_output_expected)
    assert norms["bridge_internal"] == pytest.approx(bridge_internal_expected)
    assert norms["bridge_internal"] > norms["bridge_output"]


def test_p17_bootstrap_failure_is_transactional_before_optimizer_or_scheduler_step(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    before = copy.deepcopy(trainer.state_dict())
    monkeypatch.setattr(
        trainer,
        "_rezero_norms",
        lambda: {
            "bridge_output": 0.0,
            "unary_output": 0.0,
            "pairwise_output": 0.0,
            "bridge_internal": 0.0,
            "unary_internal": 0.0,
            "pairwise_internal": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="ReZero bootstrap failed at update0"):
        trainer.train_microbatch(_batch(), epoch=0)
    after = trainer.state_dict()
    for key in (
        "model", "optimizer", "scheduler", "admission", "microbatch_step",
        "optimizer_step", "owner_optimizer_step_count", "bootstrap",
        "accumulated_owner_grads",
    ):
        _assert_nested_exact(before[key], after[key], key)


def test_p17_bootstrap_failure_rolls_back_rng_pu_buffers_and_forward_mutable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    model = _TransactionalTinyRAEL()
    trainer = module.RAELTrainer(
        model, total_optimizer_updates=20, gradient_accumulation_steps=2
    )
    trainer.set_pu_label_gate(
        torch.ones(REASON_COUNT, dtype=torch.bool),
        torch.full((REASON_COUNT,), 0.2),
    )
    trainer.train_microbatch(_batch(), epoch=1)
    assert trainer.microbatch_step == 1
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    failing_batch = _batch()
    before = copy.deepcopy(trainer.state_dict())
    before_forward_calls = model.forward_calls
    monkeypatch.setattr(
        trainer,
        "_rezero_norms",
        lambda: {
            "bridge_output": 0.0,
            "unary_output": 0.0,
            "pairwise_output": 0.0,
            "bridge_internal": 0.0,
            "unary_internal": 0.0,
            "pairwise_internal": 0.0,
        },
    )
    with pytest.raises(RuntimeError, match="ReZero bootstrap failed at update0"):
        trainer.train_microbatch(failing_batch, epoch=0)
    after = trainer.state_dict()
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "admission",
        "epoch",
        "microbatch_step",
        "optimizer_step",
        "owner_optimizer_step_count",
        "bootstrap",
        "accumulated_owner_grads",
        "python_rng",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
    ):
        _assert_nested_exact(before[key], after[key], key)
    assert model.forward_calls == before_forward_calls
    assert bool(model._pu_active_labels.all())


def test_p17_accumulation_replaces_shared_boundaries_without_erasing_owner_gradients() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=2)
    first = trainer.train_microbatch(_batch(), epoch=1)
    assert not first.optimizer_stepped
    first_grad = model.action_category.weight.grad.detach().clone()
    second = trainer.train_microbatch(_batch(), epoch=1)
    assert second.optimizer_stepped
    assert second.admission_hook_count == 2
    assert second.admission_registered_count == second.admission_triggered_count == second.admission_removed_count == 2
    assert second.owner_gradient_norms_pre_clip["action_category"] > 0.0
    assert not torch.allclose(first_grad, torch.zeros_like(first_grad))
    assert second.optimizer_step == 1


def test_p17_decay_only_optimizer_effect_is_not_reported_as_task_active() -> None:
    module = _module()
    model = _TinyRAEL()
    model.action_category.weight.register_hook(torch.zeros_like)
    model.action_category.bias.register_hook(torch.zeros_like)
    before = model.action_category.weight.detach().clone()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    result = trainer.train_microbatch(_batch(), epoch=0)
    assert result.owner_task_gradient_norms_pre_clip["action_category"] == pytest.approx(0.0)
    assert result.owner_parameter_delta["action_category"] == pytest.approx(0.0)
    assert result.owner_optimizer_effect_delta["action_category"] > 0.0
    assert result.owner_decay_only_parameter_delta["action_category"] > 0.0
    assert result.owner_optimizer_step_count["action_category"] == 0
    assert not torch.equal(before, model.action_category.weight)


def test_p17_resume_state_is_next_update_equivalent_and_has_no_off_by_one() -> None:
    module = _module()
    first = _TinyRAEL()
    second = _TinyRAEL()
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    trainer_a = module.RAELTrainer(first, total_optimizer_updates=20, gradient_accumulation_steps=2)
    trainer_b = module.RAELTrainer(second, total_optimizer_updates=20, gradient_accumulation_steps=2)
    batch = _batch()
    trainer_a.train_microbatch(batch, epoch=1)
    state = copy.deepcopy(trainer_a.state_dict())
    trainer_b.load_state_dict(state)
    result_a = trainer_a.train_microbatch(batch, epoch=1)
    result_b = trainer_b.train_microbatch(batch, epoch=1)
    assert result_a.optimizer_step == result_b.optimizer_step == 1
    assert trainer_a.microbatch_step == trainer_b.microbatch_step == 2
    assert trainer_a.optimizer_step == trainer_b.optimizer_step
    for name, value in first.state_dict().items():
        assert torch.allclose(value, second.state_dict()[name])


def test_p17_resume_restores_mid_accumulation_gradients_before_a_different_next_batch() -> None:
    """A checkpoint at microbatch 1/2 must retain the first half-gradient."""
    module = _module()
    original = _TinyRAEL()
    resumed = _TinyRAEL()
    resumed.load_state_dict(copy.deepcopy(original.state_dict()))
    trainer_a = module.RAELTrainer(original, total_optimizer_updates=20, gradient_accumulation_steps=2)
    trainer_b = module.RAELTrainer(resumed, total_optimizer_updates=20, gradient_accumulation_steps=2)
    batch_a = _batch()
    batch_b = _batch()
    batch_b["images"] = batch_b["images"] + 0.37
    trainer_a.train_microbatch(batch_a, epoch=1)
    checkpoint = copy.deepcopy(trainer_a.state_dict())
    expected_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in original.named_parameters()
        if parameter.requires_grad
    }
    assert any(gradient is not None for gradient in expected_grads.values())
    trainer_b.load_state_dict(checkpoint)
    for name, parameter in resumed.named_parameters():
        if not parameter.requires_grad:
            continue
        expected = expected_grads[name]
        if expected is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            assert torch.equal(parameter.grad, expected)

    trainer_a.train_microbatch(batch_b, epoch=1)
    trainer_b.train_microbatch(batch_b, epoch=1)
    for name, parameter in original.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, resumed.state_dict()[name], rtol=0.0, atol=1.0e-10)


def test_p17_resume_after_completed_update_preserves_full_mid_accumulation_lifecycle() -> None:
    module = _module()
    original = _RNGTinyRAEL()
    resumed_model = _RNGTinyRAEL()
    resumed_model.load_state_dict(copy.deepcopy(original.state_dict()))
    trainer_a = module.RAELTrainer(original, total_optimizer_updates=20, gradient_accumulation_steps=2)
    trainer_b = module.RAELTrainer(resumed_model, total_optimizer_updates=20, gradient_accumulation_steps=2)
    active = torch.zeros(REASON_COUNT, dtype=torch.bool)
    active[:3] = True
    trainer_a.set_pu_label_gate(active, torch.linspace(0.0, 0.2, REASON_COUNT))
    trainer_a.set_view_ema_state({"view": torch.tensor([0.25, 0.75]), "updates": 3})
    trainer_a.set_posthoc_calibration_state({"theta": torch.tensor([-0.2, 0.1]), "epoch": 1})
    batches = []
    for offset in (0.0, 0.1, 0.2, 0.3):
        item = _batch()
        item["images"] = item["images"] + offset
        batches.append(item)

    trainer_a.train_microbatch(batches[0], epoch=1)
    trainer_a.train_microbatch(batches[1], epoch=1)
    assert trainer_a.optimizer_step == 1
    trainer_a.train_microbatch(batches[2], epoch=1)
    assert trainer_a.microbatch_step == 3 and trainer_a.optimizer_step == 1
    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    checkpoint = copy.deepcopy(trainer_a.state_dict())
    assert checkpoint["optimizer"]["state"]
    assert int(checkpoint["admission"]["evidence_ema_updates"].item()) > 0
    assert any(not payload["is_none"] for payload in checkpoint["accumulated_owner_grads"].values())

    trainer_b.load_state_dict(checkpoint)
    restored = trainer_b.state_dict()
    for key in (
        "model", "optimizer", "scheduler", "admission", "pu_lambda",
        "pu_active_labels", "view_ema_state", "posthoc_calibration_state",
        "python_rng", "numpy_rng", "torch_rng", "cuda_rng",
        "accumulated_owner_grads", "owner_optimizer_step_count", "bootstrap",
        "microbatch_step", "optimizer_step",
    ):
        _assert_nested_exact(checkpoint[key], restored[key], key)

    result_a = trainer_a.train_microbatch(batches[3], epoch=1)
    state_a = copy.deepcopy(trainer_a.state_dict())
    random.setstate(checkpoint["python_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if checkpoint["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    result_b = trainer_b.train_microbatch(batches[3], epoch=1)
    state_b = trainer_b.state_dict()
    assert result_a.optimizer_step == result_b.optimizer_step == 2
    for key in (
        "model", "optimizer", "scheduler", "admission", "pu_lambda",
        "pu_active_labels", "view_ema_state", "posthoc_calibration_state",
        "accumulated_owner_grads", "owner_optimizer_step_count", "bootstrap",
        "microbatch_step", "optimizer_step", "python_rng", "numpy_rng",
        "torch_rng", "cuda_rng",
    ):
        _assert_nested_exact(state_a[key], state_b[key], f"next.{key}")


def test_p17_boundary_checkpoint_restores_no_accumulated_gradients() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    trainer.train_microbatch(_batch(), epoch=1)
    state = copy.deepcopy(trainer.state_dict())
    restored_model = _TinyRAEL()
    restored = module.RAELTrainer(restored_model, total_optimizer_updates=20, gradient_accumulation_steps=1)
    restored.load_state_dict(state)
    assert all(parameter.grad is None for parameter in restored_model.parameters() if parameter.requires_grad)


def test_p17_resume_rejects_owner_or_accumulation_configuration_mismatch() -> None:
    module = _module()
    model = _TinyRAEL()
    trainer = module.RAELTrainer(model, total_optimizer_updates=20, gradient_accumulation_steps=2)
    trainer.train_microbatch(_batch(), epoch=1)
    state = copy.deepcopy(trainer.state_dict())

    wrong_accum = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=3)
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        wrong_accum.load_state_dict(state)

    assert "owner_parameter_names" in state, "P17 checkpoints must fingerprint owner parameters"
    bad_owner_state = copy.deepcopy(state)
    bad_owner_state["owner_parameter_names"]["pu_private"] = ("reason_private.weight",)
    same_config = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="owner"):
        same_config.load_state_dict(bad_owner_state)


def test_p17_resume_round_trips_view_ema_posthoc_calibration_and_fingerprints() -> None:
    module = _module()
    trainer = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    assert hasattr(trainer, "set_view_ema_state"), "P17 must expose the view EMA state producer API"
    assert hasattr(trainer, "set_posthoc_calibration_state"), "P17 must expose the posthoc calibration state API"
    view_ema = {"evidence": torch.tensor([0.2, 0.8]), "update": 7}
    posthoc = {"theta": torch.tensor([-0.1, 0.3]), "fitted_epoch": 4}
    trainer.set_view_ema_state(view_ema)
    trainer.set_posthoc_calibration_state(posthoc)
    state = copy.deepcopy(trainer.state_dict())
    assert set(state["resume_fingerprints"]) == {
        "fingerprint_schema",
        "phase",
        "complete",
        "groups",
        "file_status",
        "file_sha256",
        "missing_files",
        "group_hashes",
        "source_hash",
        "config_hash",
        "schema_hash",
        "required_files_hash",
    }

    restored = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    restored.load_state_dict(state)
    restored_state = restored.state_dict()
    assert restored_state["resume_fingerprints"] == state["resume_fingerprints"]
    assert torch.equal(restored_state["view_ema_state"]["evidence"], view_ema["evidence"])
    assert restored_state["view_ema_state"]["update"] == 7
    assert torch.equal(restored_state["posthoc_calibration_state"]["theta"], posthoc["theta"])
    assert restored_state["posthoc_calibration_state"]["fitted_epoch"] == 4


@pytest.mark.parametrize(
    "mismatched_key",
    ("source_hash", "config_hash", "schema_hash", "required_files_hash"),
)
def test_p17_resume_rejects_each_fingerprint_mismatch(mismatched_key: str) -> None:
    module = _module()
    trainer = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    state = copy.deepcopy(trainer.state_dict())
    assert "resume_fingerprints" in state, "P17 checkpoints must carry immutable resume fingerprints"
    state["resume_fingerprints"][mismatched_key] = "a" * 64
    incompatible = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="P17 resume fingerprint manifest mismatch"):
        incompatible.load_state_dict(state)


def _independent_rael_fingerprint_groups(repository_root: Path) -> dict[str, tuple[Path, ...]]:
    """Enumerate the atomic-contract RAEL set without calling production code."""

    return {
        "source": tuple(
            sorted(
                (
                    path.relative_to(repository_root)
                    for path in (repository_root / "fate_oia").rglob("*.py")
                    if "rael" in path.name.lower()
                ),
                key=lambda path: path.as_posix(),
            )
        ),
        "test": tuple(
            sorted(
                (
                    path.relative_to(repository_root)
                    for path in (repository_root / "tests").glob("test_rael_*.py")
                ),
                key=lambda path: path.as_posix(),
            )
        ),
        "config": (Path("configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml"),),
        "schema": (
            Path("configs/rael_action_semantics.yaml"),
            Path("configs/rael_reason_semantics.yaml"),
            Path("configs/rael_slot_schema.yaml"),
        ),
        "skill": (Path(".codex/skills/rael-oia-v1-implementation-audit/SKILL.md"),),
    }


def _copy_independent_rael_fingerprint_tree(
    source_root: Path,
    destination_root: Path,
) -> dict[str, tuple[Path, ...]]:
    groups = _independent_rael_fingerprint_groups(source_root)
    for paths in groups.values():
        for relative in paths:
            source = source_root / relative
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    return groups


def _assert_checkpoint_rejects_fingerprint_change(
    module: Any,
    checkpoint: dict[str, Any],
    changed: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda: copy.deepcopy(changed),
    )
    incompatible = module.RAELTrainer(
        _TinyRAEL(),
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
        resume_fingerprints=changed,
    )
    with pytest.raises(ValueError, match="P17 resume fingerprint manifest mismatch"):
        incompatible.load_state_dict(checkpoint)


def test_p17_repository_fingerprint_manifest_is_independently_complete_and_deterministic(
    tmp_path: Path,
) -> None:
    module = _module()
    source_root = Path(module.__file__).resolve().parents[2]
    expected = _copy_independent_rael_fingerprint_tree(source_root, tmp_path)

    manifest = module.build_rael_repository_fingerprints(tmp_path)
    assert manifest["fingerprint_schema"] == "rael-repository-fingerprint-v4"
    assert manifest["phase"] == "development"
    assert set(manifest["groups"]) == {
        "source", "test", "config", "schema", "skill", "script"
    }
    assert set(manifest["group_hashes"]) == set(manifest["groups"])
    expected_paths = {
        relative.as_posix() for paths in expected.values() for relative in paths
    }
    assert expected_paths <= set(manifest["file_sha256"])
    assert set(manifest["file_status"]) == set(manifest["file_sha256"])
    assert manifest["complete"] == (not manifest["missing_files"])
    assert manifest["missing_files"] == sorted(
        path
        for path, status in manifest["file_status"].items()
        if status == "missing"
    )
    for group, paths in expected.items():
        assert {path.as_posix() for path in paths} <= set(manifest["groups"][group])
    for path, digest in manifest["file_sha256"].items():
        if manifest["file_status"][path] == "present":
            assert len(digest) == 64 and int(digest, 16) >= 0
        else:
            assert digest is None
    assert json.dumps(manifest, sort_keys=True, separators=(",", ":")) == json.dumps(
        module.build_rael_repository_fingerprints(tmp_path),
        sort_keys=True,
        separators=(",", ":"),
    )


def test_p17_repository_fingerprint_rejects_modified_deleted_and_script_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    fingerprint_builder = module.build_rael_repository_fingerprints
    source_root = Path(module.__file__).resolve().parents[2]
    independent = _copy_independent_rael_fingerprint_tree(source_root, tmp_path)
    baseline = fingerprint_builder(tmp_path)
    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda: copy.deepcopy(baseline),
    )
    owner = module.RAELTrainer(
        _TinyRAEL(),
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
        resume_fingerprints=baseline,
    )
    checkpoint = copy.deepcopy(owner.state_dict())
    target = tmp_path / independent["test"][0]
    original = target.read_bytes()

    target.write_bytes(original + b"\n# P17 R3 test mutation\n")
    changed = fingerprint_builder(tmp_path)
    _assert_checkpoint_rejects_fingerprint_change(
        module, checkpoint, changed, monkeypatch
    )
    target.write_bytes(original)

    script = tmp_path / "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text("Write-Output 'baseline'\n", encoding="utf-8")
    baseline_with_script = fingerprint_builder(tmp_path)
    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda: copy.deepcopy(baseline_with_script),
    )
    script_owner = module.RAELTrainer(
        _TinyRAEL(),
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
        resume_fingerprints=baseline_with_script,
    )
    script_checkpoint = copy.deepcopy(script_owner.state_dict())
    script.write_text("Write-Output 'changed'\n", encoding="utf-8")
    changed = fingerprint_builder(tmp_path)
    _assert_checkpoint_rejects_fingerprint_change(
        module, script_checkpoint, changed, monkeypatch
    )

    target.unlink()
    changed = fingerprint_builder(tmp_path)
    _assert_checkpoint_rejects_fingerprint_change(
        module, checkpoint, changed, monkeypatch
    )


def test_p17_stale_supplied_manifest_cannot_replace_current_repository_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    manifest_a = module.build_rael_repository_fingerprints()
    manifest_b = copy.deepcopy(manifest_a)
    first_path = next(iter(manifest_b["file_sha256"]))
    owning_group = next(
        group
        for group, paths in manifest_b["groups"].items()
        if first_path in paths
    )
    manifest_b["file_sha256"][first_path] = "a" * 64
    manifest_b["group_hashes"][owning_group] = "b" * 64
    manifest_b["source_hash"] = "c" * 64
    manifest_b["required_files_hash"] = "d" * 64
    assert manifest_b != manifest_a

    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda: copy.deepcopy(manifest_a),
    )
    owner = module.RAELTrainer(
        _TinyRAEL(),
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
        resume_fingerprints=manifest_a,
    )
    checkpoint_a = copy.deepcopy(owner.state_dict())

    monkeypatch.setattr(
        module,
        "build_rael_repository_fingerprints",
        lambda: copy.deepcopy(manifest_b),
    )
    with pytest.raises(
        ValueError,
        match="supplied fingerprint manifest does not match current repository",
    ):
        stale = module.RAELTrainer(
            _TinyRAEL(),
            total_optimizer_updates=20,
            gradient_accumulation_steps=2,
            resume_fingerprints=manifest_a,
        )
        stale.load_state_dict(checkpoint_a)


def test_p17_checkpoint_rejects_legacy_aggregate_only_fingerprints() -> None:
    module = _module()
    owner = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    checkpoint = copy.deepcopy(owner.state_dict())
    checkpoint["resume_fingerprints"] = {
        "source_hash": "a" * 64,
        "config_hash": "b" * 64,
        "schema_hash": "c" * 64,
    }
    incompatible = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="fingerprint manifest"):
        incompatible.load_state_dict(checkpoint)


def _write_fingerprint_fixture(root: Path) -> None:
    files = {
        "fate_oia/engine/train_acpr_rael_oia.py": (
            "from fate_oia.shared.direct import value\n"
            "def owner():\n"
            "    return value\n"
        ),
        "fate_oia/shared/direct.py": (
            "from fate_oia.shared.transitive import TRANSITIVE\n"
            "value = TRANSITIVE\n"
        ),
        "fate_oia/shared/transitive.py": "TRANSITIVE = 1\n",
        "tests/test_rael_train_protocol.py": "def test_fixture():\n    assert True\n",
        "configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml": "training: {}\n",
        "configs/rael_action_semantics.yaml": "actions: {}\n",
        "configs/rael_reason_semantics.yaml": "reasons: {}\n",
        "configs/rael_slot_schema.yaml": "slots: {}\n",
        ".codex/skills/rael-oia-v1-implementation-audit/SKILL.md": "# fixture\n",
        "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1": "Write-Output 'fixture'\n",
    }
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _fixture_declared_groups() -> dict[str, tuple[Path, ...]]:
    return {
        "source": (
            Path("fate_oia/engine/train_acpr_rael_oia.py"),
            Path("fate_oia/engine/eval_acpr_rael_oia.py"),
        ),
        "test": (Path("tests/test_rael_train_protocol.py"),),
        "config": (Path("configs/fate_oia_train_360x640_acpr_rael_oia_v1.yaml"),),
        "schema": (
            Path("configs/rael_action_semantics.yaml"),
            Path("configs/rael_reason_semantics.yaml"),
            Path("configs/rael_slot_schema.yaml"),
        ),
        "skill": (Path(".codex/skills/rael-oia-v1-implementation-audit/SKILL.md"),),
        "script": (Path("scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1"),),
    }


def test_f05_r2_manifest_tracks_declared_missing_files_phase_and_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _write_fingerprint_fixture(tmp_path)
    monkeypatch.setattr(module, "_RAEL_DECLARED_GROUPS", _fixture_declared_groups())

    manifest = module.build_rael_repository_fingerprints(tmp_path, phase="development")

    assert manifest["fingerprint_schema"] == "rael-repository-fingerprint-v4"
    assert manifest["phase"] == "development"
    assert manifest["complete"] is False
    missing = "fate_oia/engine/eval_acpr_rael_oia.py"
    assert manifest["missing_files"] == [missing]
    assert manifest["file_status"][missing] == "missing"
    assert manifest["file_sha256"][missing] is None
    script = "scripts/FATE_OIA_acpr_rael_oia_v1_foreground.ps1"
    assert script in manifest["groups"]["script"]
    assert manifest["file_status"][script] == "present"


def test_f05_r2_manifest_tracks_direct_and_transitive_non_rael_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _write_fingerprint_fixture(tmp_path)
    groups = _fixture_declared_groups()
    groups["source"] = (Path("fate_oia/engine/train_acpr_rael_oia.py"),)
    monkeypatch.setattr(module, "_RAEL_DECLARED_GROUPS", groups)

    baseline = module.build_rael_repository_fingerprints(tmp_path)
    direct = "fate_oia/shared/direct.py"
    transitive = "fate_oia/shared/transitive.py"
    assert direct in baseline["groups"]["source"]
    assert transitive in baseline["groups"]["source"]

    (tmp_path / direct).write_text(
        "from fate_oia.shared.transitive import TRANSITIVE\nvalue = TRANSITIVE + 1\n",
        encoding="utf-8",
    )
    direct_changed = module.build_rael_repository_fingerprints(tmp_path)
    assert direct_changed["required_files_hash"] != baseline["required_files_hash"]

    (tmp_path / direct).write_text(
        "from fate_oia.shared.transitive import TRANSITIVE\nvalue = TRANSITIVE\n",
        encoding="utf-8",
    )
    (tmp_path / transitive).write_text("TRANSITIVE = 2\n", encoding="utf-8")
    transitive_changed = module.build_rael_repository_fingerprints(tmp_path)
    assert transitive_changed["required_files_hash"] != baseline["required_files_hash"]


def test_f05_r2_missing_to_present_and_phase_changes_are_resume_incompatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    _write_fingerprint_fixture(tmp_path)
    monkeypatch.setattr(module, "_RAEL_DECLARED_GROUPS", _fixture_declared_groups())
    incomplete = module.build_rael_repository_fingerprints(tmp_path, phase="development")

    missing = tmp_path / "fate_oia/engine/eval_acpr_rael_oia.py"
    missing.write_text("VALUE = 1\n", encoding="utf-8")
    complete = module.build_rael_repository_fingerprints(tmp_path, phase="development")
    clean_head = module.build_rael_repository_fingerprints(tmp_path, phase="clean_head")

    assert incomplete["complete"] is False
    assert complete["complete"] is True
    assert complete["required_files_hash"] != incomplete["required_files_hash"]
    assert clean_head["phase"] == "clean_head"
    assert clean_head["required_files_hash"] != complete["required_files_hash"]


def test_p17_step_reports_real_admission_hook_registration_trigger_and_removal() -> None:
    module = _module()
    trainer = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1)
    result = trainer.train_microbatch(_batch(), epoch=1)
    assert result.admission_registered_count == 2
    assert result.admission_triggered_count == 2
    assert result.admission_removed_count == 2
    assert result.admission_hook_count == result.admission_triggered_count


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_p17_cuda_trainer_owns_admission_on_model_boundary_device_across_resume() -> None:
    module = _module()
    model = _TinyRAEL().to("cuda")
    device = next(model.parameters()).device
    trainer = module.RAELTrainer(
        model,
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
    )
    assert trainer.admission.state_device == device
    evidence = torch.randn(1, SLOT_COUNT, DIM, device=device, requires_grad=True)
    semantic = torch.randn(1, REASON_COUNT, DIM, device=device, requires_grad=True)
    admitted = trainer.admission.admit_from_losses(
        evidence_slots=evidence,
        semantic_reason_tokens=semantic,
        action_loss=evidence.square().mean() + semantic.square().mean(),
        reason_loss=evidence.mean() + semantic.mean(),
        grounding_loss=None,
        counterfactual_loss=None,
    )
    assert admitted.evidence.admitted is not None
    assert admitted.semantic.admitted is not None
    assert admitted.evidence.admitted.device == device
    assert admitted.semantic.admitted.device == device
    checkpoint = copy.deepcopy(trainer.state_dict())

    restored = module.RAELTrainer(
        _TinyRAEL().to(device),
        total_optimizer_updates=20,
        gradient_accumulation_steps=2,
    )
    restored.load_state_dict(checkpoint)
    assert restored.admission.state_device == device
    assert restored.admission.evidence_action_ema.device == device
    assert restored.admission.semantic_action_ema.device == device


def test_p17_requires_the_explicit_canonical_evidence_slots_key() -> None:
    module = _module()
    trainer = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1)
    outputs = trainer.model(_batch()["images"])
    outputs.pop("evidence_slots")
    with pytest.raises(ValueError, match=r"outputs\['evidence_slots'\]"):
        trainer._canonical_evidence_boundary(outputs)


def test_p17_admission_context_clears_real_hook_handles_after_backward_exception() -> None:
    module = _module()
    trainer = module.RAELTrainer(_TinyRAEL(), total_optimizer_updates=20, gradient_accumulation_steps=1)
    captured_contexts: list[Any] = []
    original_factory = trainer.admission.replace_shared_boundary_gradients

    def capture_context(**kwargs: Any):
        context = original_factory(**kwargs)
        captured_contexts.append(context)
        return context

    trainer.admission.replace_shared_boundary_gradients = capture_context  # type: ignore[method-assign]
    trainer.model.action_category.weight.register_hook(lambda _gradient: (_ for _ in ()).throw(RuntimeError("P17 injected backward failure")))
    with pytest.raises(RuntimeError, match="injected backward failure"):
        trainer.train_microbatch(_batch(), epoch=1)
    assert len(captured_contexts) == 1
    assert captured_contexts[0]._handles == []


def test_p17_counterfactual_runs_only_every_eight_optimizer_updates() -> None:
    module = _module()
    model = _TinyRAEL()
    calls: list[int] = []

    def cf_loss(_outputs: dict[str, Tensor], update: int) -> Tensor:
        calls.append(update)
        return torch.tensor(0.5, requires_grad=True)

    trainer = module.RAELTrainer(
        model,
        total_optimizer_updates=20,
        gradient_accumulation_steps=1,
        counterfactual_loss_fn=cf_loss,
    )
    for _ in range(8):
        trainer.train_microbatch(_batch(), epoch=1)
    assert calls == [8]


def test_p17_default_counterfactual_uses_formal_one_encode_replay_at_update_eight() -> None:
    module = _module()
    model = _CounterfactualTinyRAEL()
    trainer = module.RAELTrainer(
        model,
        total_optimizer_updates=20,
        gradient_accumulation_steps=1,
    )
    batch = _batch()
    results = [trainer.train_microbatch(batch, epoch=1) for _ in range(8)]

    assert trainer.optimizer_step == 8
    assert model.encode_calls == 1
    assert model.decode_calls == 1
    assert model.replay_build_calls == 1
    assert trainer.last_counterfactual_result["available"] is True
    assert bool(
        trainer.last_counterfactual_result["diagnostics"]["computed"].item()
    )
    assert results[-1].components["counterfactual"].item() > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_p17_cuda_bf16_counterfactual_replay_is_available_and_finite_at_update_eight() -> None:
    module = _module()
    device = torch.device("cuda")
    model = _CounterfactualTinyRAEL().to(device)
    trainer = module.RAELTrainer(
        model,
        total_optimizer_updates=20,
        gradient_accumulation_steps=1,
        precision="bf16",
    )

    def _to_device(value: Any) -> Any:
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: _to_device(item) for key, item in value.items()}
        return value

    batch = _to_device(_batch())
    results = [trainer.train_microbatch(batch, epoch=1) for _ in range(8)]
    result = trainer.last_counterfactual_result

    assert trainer.optimizer_step == 8
    assert model.encode_calls == 1
    assert model.decode_calls == 1
    assert model.replay_build_calls == 1
    assert result["available"] is True
    assert result["reason"] == "ok"
    assert bool(result["diagnostics"]["computed"].item())
    assert torch.isfinite(results[-1].components["counterfactual"])
    assert results[-1].components["counterfactual"].item() > 0.0
    assert all(
        torch.isfinite(value).all()
        for value in result["diagnostics"].values()
        if isinstance(value, Tensor) and value.dtype != torch.bool
    )


@pytest.mark.parametrize("gradient_accumulation_steps", (2, 3))
def test_p17_counterfactual_only_runs_on_the_eighth_completed_update_boundary(
    gradient_accumulation_steps: int,
) -> None:
    """No step-zero CF and no duplicate CF after update eight."""
    module = _module()
    model = _TinyRAEL()
    calls: list[int] = []

    def cf_loss(_outputs: dict[str, Tensor], update: int) -> Tensor:
        calls.append(update)
        return torch.tensor(0.5, requires_grad=True)

    trainer = module.RAELTrainer(
        model,
        total_optimizer_updates=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        counterfactual_loss_fn=cf_loss,
    )
    batch = _batch()
    # The first non-boundary microbatch is the historical failure: old code
    # treated optimizer_step=0 as an event because 0 % 8 == 0.
    trainer.train_microbatch(batch, epoch=1)
    assert calls == []

    # Complete exactly eight optimizer updates.  Only the final boundary may
    # schedule counterfactual work, and it must use completed update id 8.
    for _ in range(gradient_accumulation_steps * 8 - 1):
        trainer.train_microbatch(batch, epoch=1)
    assert trainer.optimizer_step == 8
    assert calls == [8]

    # The next non-boundary microbatch must not replay update eight's CF.
    trainer.train_microbatch(batch, epoch=1)
    assert trainer.optimizer_step == 8
    assert calls == [8]


def test_p17_rezero_bootstrap_enforces_update_zero_one_two_deadlines() -> None:
    module = _module()
    tracker = module.ReZeroBootstrapTracker()
    tracker.observe(0, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 0.0, "bridge_internal": 0.0, "unary_internal": 0.0, "pairwise_internal": 0.0})
    tracker.observe(1, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 1.0, "bridge_internal": 1.0, "unary_internal": 1.0, "pairwise_internal": 0.0})
    tracker.observe(2, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 1.0, "bridge_internal": 1.0, "unary_internal": 1.0, "pairwise_internal": 1.0})
    tracker.assert_satisfied()
    with pytest.raises(RuntimeError, match="pairwise"):
        failed = module.ReZeroBootstrapTracker()
        failed.observe(0, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 0.0, "bridge_internal": 0.0, "unary_internal": 0.0, "pairwise_internal": 0.0})
        failed.observe(1, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 0.0, "bridge_internal": 1.0, "unary_internal": 1.0, "pairwise_internal": 0.0})
        failed.observe(2, {"bridge_output": 1.0, "unary_output": 1.0, "pairwise_output": 0.0, "bridge_internal": 1.0, "unary_internal": 1.0, "pairwise_internal": 0.0})
        failed.assert_satisfied()


def test_p18_observes_the_real_reason_private_rezero_parameter() -> None:
    source = inspect.getsource(_module().RAELTrainer._mechanism_observation)
    assert '"gamma_RA": ("reason_private", "gamma_ra_raw")' in source
    assert "gamma_private_raw" not in source
