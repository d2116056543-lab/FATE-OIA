from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from fate_oia.engine.mosaic_icdor_audit_collectors import (
    collect_edge_intervention_audit,
    collect_factor_audit,
)


class _Grounding:
    def __call__(self, records, *, device: torch.device, split: str):
        assert split == "train"
        assert len(records) == 4
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], device=device)
        masks = torch.zeros(4, 2, 8, 8, device=device)
        masks[:2, 0, 0, 0] = 1.0
        masks[1, 0, 0, 0] = 0.0
        masks[1, 0, 0, 1] = 1.0
        masks[2:, 1, 6, 6] = 1.0
        masks[3, 1, 6, 6] = 0.0
        masks[3, 1, 6, 5] = 1.0
        return {
            "presence_target": target,
            "presence_known_mask": torch.ones_like(target),
            "visibility_target": target,
            "visibility_known_mask": torch.ones_like(target),
            "weak_negative_mask": torch.zeros_like(target),
            "geometry_known_mask": target,
            "geometry_masks": masks,
        }


class _AuditModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled_during_forward: list[bool] = []
        self.action_router = SimpleNamespace(
            candidate_edge_mask=torch.ones(2, 2, 2, dtype=torch.bool),
            edge_admission_mask=torch.tensor(
                [[[True, False], [False, False]], [[False, False], [False, False]]], dtype=torch.bool
            ),
        )

    def set_edge_admission(self, mask: torch.Tensor) -> None:
        self.action_router.edge_admission_mask = mask.detach().clone()

    def forward(self, images: torch.Tensor, *, factor_ablation_mode: str = "full", **_) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        probabilities = {
            "full": [[0.95, 0.05], [0.85, 0.10], [0.10, 0.80], [0.05, 0.90]],
            "content_only": [[0.90, 0.10], [0.75, 0.15], [0.15, 0.75], [0.10, 0.85]],
            "prior_only": [[0.40, 0.60], [0.45, 0.55], [0.55, 0.45], [0.60, 0.40]],
            "query_shuffled": [[0.10, 0.90], [0.15, 0.85], [0.85, 0.15], [0.90, 0.10]],
            "image_shuffled": [[0.25, 0.70], [0.30, 0.65], [0.70, 0.30], [0.75, 0.25]],
        }[factor_ablation_mode]
        probability = torch.tensor(probabilities, device=images.device)
        masks = torch.zeros(4, 2, 8, 8, device=images.device)
        masks[:, 0, 0, 0] = 1.0
        masks[1, 0, 0, 0] = 0.0
        masks[1, 0, 0, 1] = 1.0
        masks[2:, 1, 6, 6] = 1.0
        masks[3, 1, 6, 6] = 0.0
        masks[3, 1, 6, 5] = 1.0
        # Every selected factor needs an independent same-type identity arm.
        masks[:, 1, 4, 4] = 1.0
        override = _.get("factor_mask_override")
        if isinstance(override, torch.Tensor):
            masks = override.to(images.device)
        if float(images.mean()) > 0.5:
            masks = torch.flip(masks, dims=(-1,))
        edge_gain = self.action_router.edge_admission_mask[0, 0, 0].float()
        random_gain = self.action_router.edge_admission_mask[0, 1, 0].float()
        action_logits = torch.tensor([[2.0, -1.0], [1.5, -1.0], [-1.0, 2.0], [-1.0, 1.5]], device=images.device)
        action_logits[:, 0] += 2.0 * edge_gain + 0.25 * random_gain
        action_logits[:, 0] += 0.25 * masks[:, 0, 0, 0]
        return {
            "factor_presence_prob": probability,
            "factor_visibility_prob": probability,
            "factor_soft_masks": masks,
            "factor_positive_evidence": probability,
            "factor_negative_evidence": 1.0 - probability,
            "prototype_weights": torch.tensor(
                [[[0.7, 0.3], [0.6, 0.4]]] * 4, device=images.device
            ),
            "measurement_stats": {
                "prototype_effective_count": torch.tensor([1.8, 1.9], device=images.device),
                "dominant_prototype_rate": torch.tensor([0.0, 0.0], device=images.device),
                "dead_prototype_count": torch.tensor([0.0, 0.0], device=images.device),
            },
            "action_final_logits": action_logits,
        }


def _batch(split: str = "train_audit") -> dict[str, object]:
    return {
        "split": [split] * 4,
        "image": torch.zeros(4, 3, 8, 8),
        "grounding_records": [{"id": index} for index in range(4)],
        "audit_views": torch.zeros(4, 2, 3, 8, 8),
        "audit_mirror_view": torch.ones(4, 3, 8, 8),
        "action": torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
        "reason": torch.tensor([[0.0], [0.0], [1.0], [0.0]]),
    }


class ICDORAuditCollectorsTest(unittest.TestCase):
    def test_factor_collector_uses_same_reason_positive_anchors_as_training(self) -> None:
        result = collect_factor_audit(
            _AuditModel(), [_batch()], _Grounding(),
            factor_names=("f0", "f1"),
            factor_definitions=({"name": "f0", "positive_reason_anchors": [0]}, {"name": "f1"}),
            device=torch.device("cpu"), bootstrap_replicates=16, bootstrap_seed=7,
        )
        record = result["factor_stats"]["f0"]
        self.assertEqual(record["counts"]["confirmed_positive"], 3)
        self.assertEqual(record["counts"]["reliable_negative"], 1)

    def test_factor_collector_uses_real_ablations_grounding_views_and_bootstrap(self) -> None:
        model = _AuditModel()
        result = collect_factor_audit(
            model,
            [_batch()],
            _Grounding(),
            factor_names=("f0", "f1"),
            device=torch.device("cpu"),
            bootstrap_replicates=64,
            bootstrap_seed=17,
        )

        self.assertTrue(model.grad_enabled_during_forward)
        self.assertFalse(any(model.grad_enabled_during_forward))

        self.assertEqual(result["source_split"], "train_audit")
        self.assertEqual(
            result["audit_integrity"],
            {
                "collector_completed": True,
                "exception": None,
                "unknown_policy": "excluded_from_binary_metrics",
                "unknown_rows_total": 0,
                "unknown_rows_in_metric_total": 0,
            },
        )
        record = result["factor_stats"]["f0"]
        self.assertEqual(record["counts"]["confirmed_positive"], 2)
        self.assertGreater(record["scores"]["full"], record["scores"]["prior_only"])
        self.assertGreater(record["scores"]["query_shuffle_drop"], 0.0)
        self.assertGreater(record["scores"]["image_shuffle_drop"], 0.0)
        self.assertGreater(record["scores"]["grounding_minus_random"], 0.0)
        self.assertAlmostEqual(record["scores"]["view_consistency"], 1.0)
        self.assertGreater(record["prototype"]["effective_count"], 1.0)
        self.assertEqual(
            set(record["bootstrap_lcb95"]),
            {"full_minus_prior_only", "query_shuffle_drop", "image_shuffle_drop", "grounding_minus_random", "stability"},
        )
        self.assertEqual(
            set(record["bootstrap_positive_rate"]),
            {"full_minus_prior_only", "query_shuffle_drop", "image_shuffle_drop", "grounding_minus_random", "stability"},
        )
        self.assertTrue(all(0.0 <= value <= 1.0 for value in record["bootstrap_positive_rate"].values()))

    def test_factor_collector_proves_unknown_rows_are_excluded(self) -> None:
        class _UnknownGrounding(_Grounding):
            def __call__(self, records, *, device: torch.device, split: str):
                result = super().__call__(records, device=device, split=split)
                result["presence_known_mask"][3, 0] = 0.0
                return result

        result = collect_factor_audit(
            _AuditModel(), [_batch()], _UnknownGrounding(),
            factor_names=("f0", "f1"), device=torch.device("cpu"),
            bootstrap_replicates=16, bootstrap_seed=7,
        )
        integrity = result["audit_integrity"]
        self.assertEqual(integrity["unknown_rows_total"], 1)
        self.assertEqual(integrity["unknown_rows_in_metric_total"], 0)
        self.assertEqual(integrity["unknown_policy"], "excluded_from_binary_metrics")

    def test_collectors_reject_test_input_and_edge_collector_uses_matched_interventions(self) -> None:
        with self.assertRaisesRegex(ValueError, "train_audit"):
            collect_factor_audit(
                _AuditModel(), [_batch("test")], _Grounding(), factor_names=("f0", "f1"), device=torch.device("cpu")
            )

        result = collect_edge_intervention_audit(
            _AuditModel(),
            [_batch()],
            factor_names=("f0", "f1"),
            action_names=("stop", "go"),
            edge_specs=({"factor": "f0", "action": "stop", "direction": "support", "polarity": "present"},),
            device=torch.device("cpu"),
            bootstrap_replicates=64,
            bootstrap_seed=17,
        )

        self.assertEqual(result["source_split"], "train_audit")
        edge = result["edge_stats"]["support:f0->stop"]
        self.assertEqual(edge["matched_counts"], {
            "factor_on": 4,
            "factor_off": 4,
            "equal_mass_random": 4,
            "same_type_identity": 4,
            "spatial_roll": 4,
        })
        self.assertGreater(edge["metrics"]["tet"], edge["metrics"]["tes"])
        self.assertGreaterEqual(edge["metrics"]["tes"], 0.0)
        self.assertGreaterEqual(edge["metrics"]["tes_identity"], 0.0)
        self.assertGreaterEqual(edge["metrics"]["tes_spatial"], 0.0)
        identity_arm = edge["matched_control_arms"][0]
        self.assertEqual(identity_arm["control_type"], "same_type_identity")
        self.assertEqual(identity_arm["identity_source_factor_names"], ["f1"])
        self.assertTrue(all(arm["spatial_offsets"] for arm in edge["matched_control_arms"][1:]))
        self.assertGreaterEqual(edge["metrics"]["ap_delta"], 0.0)
        self.assertGreaterEqual(edge["metrics"]["isolated_edge_ap"], edge["metrics"]["visual_ap"])
        self.assertEqual(
            set(edge["bootstrap_ci95"]),
            {
                "signed_effect", "tet", "tes", "tes_identity", "tes_spatial",
                "cca", "ap_delta", "isolated_edge_ap", "visual_ap",
            },
        )
