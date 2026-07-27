from __future__ import annotations

import copy
import gc
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any
import weakref

import pytest
import torch
import yaml
from torch import Tensor, nn

from fate_oia.losses.rael_pu_losses import canonicalize_sample_id
from fate_oia.engine.export_rael_cases import RAELCaseExportCollector
from fate_oia.models.rael_oia_model import BRANCH_NAMES
from fate_oia.utils.rael_posthoc_calibration import fit_posthoc_calibration
from fate_oia.utils.rael_schema import load_reason_semantic_schema


EXPECTED_BRANCH_NAMES = (
    "global_only",
    "global_plus_semantic_bridge",
    "unary_only",
    "pairwise_only",
    "full",
    "no_semantic_reason",
    "semantic_reason_shuffled",
    "reason_private_shuffled",
    "named_slots_only",
    "latent_slots_only",
    "global_context_only",
    "evidence_shuffled",
    "pairwise_off",
    "pu_off",
)

# Expected names are independently fixed by the formal repository schemas.
ACTION_NAMES = ("forward", "stop", "left", "right")
REASON_NAMES = (
    "Traffic light is green",
    "Follow traffic",
    "Road is clear",
    "Traffic light",
    "Traffic sign",
    "Obstacle: car",
    "Obstacle: person",
    "Obstacle: rider",
    "Obstacle: others",
    "No lane on the left",
    "Obstacles on the left lane",
    "Solid line on the left",
    "On the left-turn lane",
    "Traffic light allows left",
    "Front car turning left",
    "No lane on the right",
    "Obstacles on the right lane",
    "Solid line on the right",
    "On the right-turn lane",
    "Traffic light allows right",
    "Front car turning right",
)


def _module():
    try:
        return importlib.import_module("fate_oia.engine.eval_acpr_rael_oia")
    except ModuleNotFoundError as error:
        pytest.fail(f"P19 RED: evaluator is absent: {error}")


class _EvalModel(nn.Module):
    def __init__(
        self,
        *,
        omit_branch: str | None = None,
        mismatch_full: bool = False,
        dino_call_count: int | Tensor | None = 1,
        decode_raises: bool = False,
        branch_insertion_order: str = "canonical",
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encode_calls = 0
        self.decode_calls = 0
        self.forward_calls = 0
        self.omit_branch = omit_branch
        self.mismatch_full = mismatch_full
        self.dino_call_count = dino_call_count
        self.decode_raises = decode_raises
        self.branch_insertion_order = branch_insertion_order

    def forward(self, *_: Any, **__: Any) -> Tensor:
        self.forward_calls += 1
        raise AssertionError("evaluator must not call model.forward")

    def encode_images(self, images: Tensor) -> dict[str, Tensor]:
        self.encode_calls += 1
        return {"images": images}

    def decode_from_field(
        self,
        field: dict[str, Tensor],
        *,
        diagnostic_modes: tuple[str, ...],
    ) -> dict[str, object]:
        self.decode_calls += 1
        if self.decode_raises:
            raise RuntimeError("injected decode failure")
        assert diagnostic_modes == EXPECTED_BRANCH_NAMES
        images = field["images"]
        action = images[:, :4].float()
        reason = torch.cat(
            [images.float(), images.float(), images[:, :5].float()], dim=1
        )
        names = list(BRANCH_NAMES)
        if self.branch_insertion_order == "full_first":
            names = ["full", *[name for name in names if name != "full"]]
        elif self.branch_insertion_order == "reversed":
            names.reverse()
        elif self.branch_insertion_order == "randomized":
            priority = ("pairwise_off", "global_only", "pu_off", "full")
            names = list(priority) + [
                name for name in reversed(BRANCH_NAMES) if name not in priority
            ]
        elif self.branch_insertion_order != "canonical":
            raise ValueError("invalid fake branch insertion order")
        branches: dict[str, dict[str, Tensor]] = {}
        for name in names:
            if name == self.omit_branch:
                continue
            index = BRANCH_NAMES.index(name)
            if name == "full":
                delta = 0.25 if self.mismatch_full else 0.0
            else:
                delta = 0.01 * (index + 1)
            branches[name] = {"action": action + delta, "reason": reason - delta}
        diagnostics: dict[str, object] = {}
        if self.dino_call_count is not None:
            diagnostics["dino_call_count"] = self.dino_call_count
        return {
            "action_logits_final": action,
            "reason_logits_final": reason,
            "branch_logits": branches,
            "diagnostics": diagnostics,
        }


def _pairwise_incident(pairwise: Tensor, pair_indices: Tensor) -> Tensor:
    incident = pairwise.new_zeros((*pairwise.shape[:2], 20))
    for side in (0, 1):
        incident.scatter_add_(
            2,
            pair_indices[:, side].view(1, 1, -1).expand_as(pairwise),
            pairwise,
        )
    return incident


class _CaseExportEvalModel(_EvalModel):
    """Adds the formal P18 fields to the normal evaluator model outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.case_output_ids: list[int] = []

    def decode_from_field(
        self,
        field: dict[str, Tensor],
        *,
        diagnostic_modes: tuple[str, ...],
    ) -> dict[str, object]:
        outputs = super().decode_from_field(field, diagnostic_modes=diagnostic_modes)
        action = outputs["action_logits_final"]
        reason = outputs["reason_logits_final"]
        assert isinstance(action, Tensor) and isinstance(reason, Tensor)
        batch_size = action.shape[0]
        slots = 20
        pairs = slots * (slots - 1) // 2
        pair_indices = torch.triu_indices(slots, slots, offset=1).transpose(0, 1).contiguous()
        action_unary = (
            0.002
            + torch.arange(slots, dtype=action.dtype, device=action.device).view(1, 1, slots)
            * 0.0001
        ).expand(batch_size, 4, slots)
        reason_unary = (
            0.001
            + torch.arange(slots, dtype=reason.dtype, device=reason.device).view(1, 1, slots)
            * 0.0001
        ).expand(batch_size, 21, slots)
        action_pairwise = (
            0.00001
            + torch.arange(pairs, dtype=action.dtype, device=action.device).view(1, 1, pairs)
            * 0.0000001
        ).expand(batch_size, 4, pairs)
        reason_pairwise = (
            0.00002
            + torch.arange(pairs, dtype=reason.dtype, device=reason.device).view(1, 1, pairs)
            * 0.0000001
        ).expand(batch_size, 21, pairs)
        action_incident = _pairwise_incident(action_pairwise, pair_indices)
        reason_incident = _pairwise_incident(reason_pairwise, pair_indices)
        masks = torch.linspace(
            0.001,
            1.0,
            batch_size * slots * 45 * 80,
            dtype=action.dtype,
            device=action.device,
        ).reshape(batch_size, slots, 45, 80)
        outputs.update(
            {
                "slot_masks": masks,
                "slot_area": masks.mean(dim=(-2, -1)),
                "slot_centroid": torch.stack(
                    [
                        torch.full((batch_size, slots), 0.4, dtype=action.dtype, device=action.device),
                        torch.full((batch_size, slots), 0.6, dtype=action.dtype, device=action.device),
                    ],
                    dim=-1,
                ),
                "slot_scale": torch.full((batch_size, slots), 0.2, dtype=action.dtype, device=action.device),
                "slot_activity": torch.full((batch_size, slots), 0.9, dtype=action.dtype, device=action.device),
                "slot_presence": torch.full((batch_size, slots), 0.8, dtype=action.dtype, device=action.device),
                "slot_observability": torch.full((batch_size, slots), 0.75, dtype=action.dtype, device=action.device),
                "slot_reliability": torch.full((batch_size, slots), 0.7, dtype=action.dtype, device=action.device),
                "slot_type_probs": torch.full((batch_size, 12, 6), 1.0 / 6.0, dtype=action.dtype, device=action.device),
                "slot_state_probs": torch.full((batch_size, 12, 4), 0.25, dtype=action.dtype, device=action.device),
                "slot_sector_probs": {
                    "horizontal": torch.full((batch_size, slots, 3), 1.0 / 3.0, dtype=action.dtype, device=action.device),
                    "depth": torch.full((batch_size, slots, 3), 1.0 / 3.0, dtype=action.dtype, device=action.device),
                },
                "action_global_contribution": action - action_unary.sum(dim=-1) - action_pairwise.sum(dim=-1),
                "reason_global_contribution": reason - reason_unary.sum(dim=-1) - reason_pairwise.sum(dim=-1),
                "action_unary_contributions": action_unary,
                "reason_unary_contributions": reason_unary,
                "action_pairwise_contributions": action_pairwise,
                "reason_pairwise_contributions": reason_pairwise,
                "action_pairwise_incident_contributions": action_incident,
                "reason_pairwise_incident_contributions": reason_incident,
                "action_analytical_deletion": action_unary + action_incident,
                "reason_analytical_deletion": reason_unary + reason_incident,
                "action_pair_indices": pair_indices,
                "reason_pair_indices": pair_indices.clone(),
            }
        )
        self.case_output_ids.append(id(outputs))
        return outputs


class _RecordingCaseCollector(RAELCaseExportCollector):
    def __init__(self) -> None:
        super().__init__(max_failure_cases=3, max_evidence_cases=4, top_slots=2)
        self.consume_calls = 0
        self.finalize_calls = 0
        self.output_ids: list[int] = []

    def consume(self, **kwargs: Any) -> None:
        self.consume_calls += 1
        self.output_ids.append(id(kwargs["outputs"]))
        super().consume(**kwargs)

    def finalize(self, provenance: dict[str, object]) -> dict[str, list[dict[str, object]]]:
        self.finalize_calls += 1
        return super().finalize(provenance)


class _ExplodingProtocolCollector:
    def consume(
        self,
        *,
        file_names: object,
        action_targets: object,
        reason_targets: object,
        outputs: object,
        action_calibration: object,
        reason_calibration: object,
        action_names: object,
        reason_names: object,
    ) -> None:
        raise RuntimeError("injected collector failure")

    def finalize(self, provenance: object) -> object:
        raise AssertionError("finalize must not follow a failed consume")


def _batch(start: int, count: int = 4) -> dict[str, object]:
    rows = torch.arange(start, start + count, dtype=torch.float32).view(-1, 1)
    images = torch.sin(rows + torch.arange(8, dtype=torch.float32).view(1, -1))
    action = ((rows + torch.arange(4).view(1, -1)) % 2 == 0).float()
    reason = ((rows + torch.arange(21).view(1, -1)) % 3 == 0).float()
    return {
        "split": "test",
        "images": images,
        "action_targets": action,
        "reason_targets": reason,
        "file_names": tuple(
            f"E:/bdd-oia/test/test-{start + index:04d}.jpg" for index in range(count)
        ),
    }


def _calib_ids(*, offset: int = 0) -> tuple[str, ...]:
    return tuple(
        f"E:/bdd-oia/train_calib/calib-{index + offset:04d}.jpg"
        for index in range(24)
    )


def _p12_split_hash(values: list[str] | tuple[str, ...]) -> str:
    canonical = [canonicalize_sample_id(value) for value in values]
    return hashlib.sha256(
        json.dumps(canonical, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _batch_file_names(batches: list[dict[str, object]]) -> tuple[str, ...]:
    return tuple(
        name
        for batch in batches
        for name in batch["file_names"]  # type: ignore[index,union-attr]
    )


class _TrackedBatch(dict[str, object]):
    """Weak-referenceable batch container for the streaming retention contract."""


class _OneShotTrackedBatches:
    def __init__(self, starts: tuple[int, ...] = (0, 4)) -> None:
        self.starts = starts
        self.iterations = 0
        self.released_batches = 0

    def __iter__(self):
        if self.iterations:
            raise AssertionError("evaluator must consume the batch iterable exactly once")
        self.iterations += 1
        for start in self.starts:
            batch = _TrackedBatch(_batch(start))
            batch_ref = weakref.ref(batch)
            image_ref = weakref.ref(batch["images"])
            yield batch
            del batch
            gc.collect()
            assert batch_ref() is None, "evaluator retained a completed batch"
            assert image_ref() is None, "evaluator retained a completed image tensor"
            self.released_batches += 1


def _schema_paths() -> tuple[Path, Path]:
    for root in (Path.cwd(), *Path(__file__).resolve().parents):
        action = root / "configs" / "rael_action_semantics.yaml"
        reason = root / "configs" / "rael_reason_semantics.yaml"
        if action.is_file() and reason.is_file():
            return action, reason
    pytest.fail("P19 evaluator tests require repository RAEL action/reason schemas")


def _calibration(
    targets: int, *, stable_ids: tuple[str, ...] | None = None
) -> dict[str, object]:
    rows = 24
    logits = torch.linspace(-1.2, 1.2, rows * targets).reshape(rows, targets)
    labels = (
        (torch.arange(rows).view(-1, 1) + torch.arange(targets).view(1, -1)) % 3
        == 0
    ).float()
    return fit_posthoc_calibration(
        raw_logits=logits.float(),
        labels=labels,
        split="train_calib",
        group_ids=[index % 4 for index in range(targets)],
        stable_ids=list(_calib_ids() if stable_ids is None else stable_ids),
    )


def _evaluate(
    *,
    model: nn.Module | None = None,
    batches: list[dict[str, object]] | None = None,
    action_calibration: dict[str, object] | None = None,
    reason_calibration: dict[str, object] | None = None,
    expected_train_calib_split_hash: str | None = None,
    expected_test_split_hash: str | None = None,
    action_schema_path: Path | str | None = None,
    reason_schema_path: Path | str | None = None,
    case_collector: object = None,
    case_export_provenance: object = None,
) -> dict[str, Any]:
    resolved_batches = [_batch(0), _batch(4)] if batches is None else batches
    default_action_schema, default_reason_schema = _schema_paths()
    case_export_kwargs: dict[str, object] = {}
    if case_collector is not None:
        case_export_kwargs["case_collector"] = case_collector
    if case_export_provenance is not None:
        case_export_kwargs["case_export_provenance"] = case_export_provenance
    return _module().evaluate_rael_test_only(
        model=_EvalModel() if model is None else model,
        batches=resolved_batches,
        action_calibration=_calibration(4) if action_calibration is None else action_calibration,
        reason_calibration=_calibration(21) if reason_calibration is None else reason_calibration,
        expected_train_calib_split_hash=(
            _p12_split_hash(_calib_ids())
            if expected_train_calib_split_hash is None
            else expected_train_calib_split_hash
        ),
        expected_test_split_hash=(
            _p12_split_hash(_batch_file_names(resolved_batches))
            if expected_test_split_hash is None
            else expected_test_split_hash
        ),
        action_schema_path=default_action_schema if action_schema_path is None else action_schema_path,
        reason_schema_path=default_reason_schema if reason_schema_path is None else reason_schema_path,
        device=torch.device("cpu"),
        **case_export_kwargs,
    )


def _assert_metric_bundle(bundle: dict[str, Any]) -> None:
    assert set(bundle) == {
        "mF1",
        "oF1",
        "mAP",
        "AUC",
        "ranking_source",
        "decision_source",
    }
    assert bundle["ranking_source"] == "raw_logits"
    assert bundle["decision_source"] in {
        "raw_zero_threshold",
        "p15_train_calib_posthoc",
    }
    assert all(math.isfinite(float(bundle[key])) for key in ("mF1", "oF1", "mAP", "AUC"))


def _assert_label_rows(payload: dict[str, Any], *, expected: int) -> None:
    assert set(payload) == {"rows"}
    rows = payload["rows"]
    assert len(rows) == expected
    for index, row in enumerate(rows):
        assert set(row) == {"id", "name", "F1", "AP", "AUC", "support", "threshold"}
        assert row["id"] == index
        assert isinstance(row["name"], str) and row["name"]
        assert all(math.isfinite(float(row[key])) for key in ("F1", "AP", "AUC", "support", "threshold"))


def _p18_artifact_module_or_none() -> Any | None:
    try:
        module = importlib.import_module("fate_oia.utils.rael_artifacts")
    except ModuleNotFoundError:
        return None
    for name in ("_validate_epoch_json", "_validate_logits_tensor", "_validate_labels_tensor"):
        if not callable(getattr(module, name, None)):
            pytest.fail(f"P18 artifact module is present but missing strict validator {name}")
    return module


def _p18_provenance(*, sample_count: int) -> dict[str, object]:
    return {
        "schema_version": "rael-artifact-v1",
        "producer": "fate_oia.engine.eval_acpr_rael_oia:p19-contract",
        "source_fingerprint_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "epoch": 0,
        "sample_count": sample_count,
    }


def test_p19_success_is_exact_p18_schema_with_one_encode_and_full_anchor() -> None:
    module = _module()
    assert tuple(BRANCH_NAMES) == EXPECTED_BRANCH_NAMES
    assert len(BRANCH_NAMES) == 14
    model = _EvalModel(dino_call_count=torch.tensor(1))
    model.train()
    result = _evaluate(model=model)
    action_schema_path, reason_schema_path = _schema_paths()
    action_schema = yaml.safe_load(action_schema_path.read_text(encoding="utf-8"))
    assert [row["name"] for row in action_schema["actions"]] == list(ACTION_NAMES)
    assert [row.name for row in load_reason_semantic_schema(reason_schema_path)] == list(
        REASON_NAMES
    )

    assert model.training is True
    assert model.encode_calls == 2
    assert model.decode_calls == 2
    assert model.forward_calls == 0
    assert result["case_exports"] is None
    assert result["selection"] == {
        "split": "test",
        "metric": "deploy_fixed_joint",
        "internal_test_selected": True,
        "publication_eligible": False,
    }
    assert result["primary_best_value"] == result["deploy_metrics"]["metrics"]["joint"]
    assert len(result["file_names"]) == 8
    assert len(set(result["file_names"])) == 8

    for key in ("raw_metrics", "deploy_metrics"):
        assert set(result[key]) == {"metrics"}
        metrics = result[key]["metrics"]
        assert set(metrics) == {"action", "reason", "joint"}
        _assert_metric_bundle(metrics["action"])
        _assert_metric_bundle(metrics["reason"])
        assert math.isfinite(float(metrics["joint"]))
    assert result["deploy_metrics"]["metrics"]["action"]["mAP"] == result["raw_metrics"]["metrics"]["action"]["mAP"]
    assert result["deploy_metrics"]["metrics"]["action"]["AUC"] == result["raw_metrics"]["metrics"]["action"]["AUC"]
    assert result["deploy_metrics"]["metrics"]["reason"]["mAP"] == result["raw_metrics"]["metrics"]["reason"]["mAP"]
    assert result["deploy_metrics"]["metrics"]["reason"]["AUC"] == result["raw_metrics"]["metrics"]["reason"]["AUC"]
    assert all(
        result["raw_metrics"]["metrics"][family]["decision_source"]
        == "raw_zero_threshold"
        for family in ("action", "reason")
    )
    assert all(
        result["deploy_metrics"]["metrics"][family]["decision_source"]
        == "p15_train_calib_posthoc"
        for family in ("action", "reason")
    )

    branches = result["branch_metrics"]
    assert set(branches) == {"branches"}
    assert [branch["name"] for branch in branches["branches"]] == list(EXPECTED_BRANCH_NAMES)
    for branch in branches["branches"]:
        assert branch["config"] == {"diagnostic_mode": branch["name"]}
        assert set(branch["metrics"]) == {"action", "reason", "joint"}
        _assert_metric_bundle(branch["metrics"]["action"])
        _assert_metric_bundle(branch["metrics"]["reason"])
        _assert_label_rows({"rows": branch["per_action"]}, expected=4)
        _assert_label_rows({"rows": branch["per_reason"]}, expected=21)
        assert [row["name"] for row in branch["per_action"]] == list(ACTION_NAMES)
        assert [row["name"] for row in branch["per_reason"]] == list(REASON_NAMES)

    _assert_label_rows(result["per_action"], expected=4)
    _assert_label_rows(result["per_reason"], expected=21)
    assert [row["name"] for row in result["per_action"]["rows"]] == list(ACTION_NAMES)
    assert [row["name"] for row in result["per_reason"]["rows"]] == list(REASON_NAMES)
    assert all(row["threshold"] == 0.0 for row in branches["branches"][4]["per_action"])
    action_chosen = _calibration(4)["chosen"]
    reason_chosen = _calibration(21)["chosen"]
    for row, threshold, temperature in zip(result["per_action"]["rows"], action_chosen["threshold"], action_chosen["temperature"]):
        assert row["threshold"] == pytest.approx(float(threshold) * float(temperature))
    for row, threshold, temperature in zip(result["per_reason"]["rows"], reason_chosen["threshold"], reason_chosen["temperature"]):
        assert row["threshold"] == pytest.approx(float(threshold) * float(temperature))

    tensors = result["tensors"]
    for family in ("action", "reason"):
        raw = tensors["logits_raw"][family]
        deploy = tensors["logits_deploy"][family]
        labels = tensors["labels"][family]
        expected = 4 if family == "action" else 21
        for tensor in (raw, deploy, labels):
            assert tensor.device.type == "cpu"
            assert tensor.requires_grad is False
            assert bool(torch.isfinite(tensor).all())
            assert tensor.shape == (8, expected)
        assert raw.dtype == torch.float32
        assert deploy.dtype == torch.float32
        assert labels.dtype == torch.float32

    artifact_module = _p18_artifact_module_or_none()
    provenance = _p18_provenance(sample_count=8)
    if artifact_module is not None:
        artifact_module._validate_epoch_json("raw_metrics.json", {**provenance, **result["raw_metrics"]}, epoch=0)
        artifact_module._validate_epoch_json("deploy_metrics.json", {**provenance, **result["deploy_metrics"]}, epoch=0)
        artifact_module._validate_epoch_json("branch_metrics.json", {**provenance, **result["branch_metrics"]}, epoch=0)
        artifact_module._validate_epoch_json("per_action.json", {**provenance, **result["per_action"]}, epoch=0)
        artifact_module._validate_epoch_json("per_reason.json", {**provenance, **result["per_reason"]}, epoch=0)
        artifact_module._validate_logits_tensor(
            "logits_raw.pt", {"_meta": provenance, **result["tensors"]["logits_raw"]}, epoch=0
        )
        artifact_module._validate_logits_tensor(
            "logits_deploy.pt", {"_meta": provenance, **result["tensors"]["logits_deploy"]}, epoch=0
        )
        artifact_module._validate_labels_tensor(
            {"_meta": provenance, **result["tensors"]["labels"], "file_names": result["file_names"]}, epoch=0
        )


def test_p19_streams_case_exports_without_recomputing_model_outputs() -> None:
    artifact_module = _p18_artifact_module_or_none()
    if artifact_module is None:
        pytest.fail("P19 case-export integration requires P18 artifact validators")
    model = _CaseExportEvalModel()
    collector = _RecordingCaseCollector()
    result = _evaluate(
        model=model,
        case_collector=collector,
        case_export_provenance=_p18_provenance(sample_count=8),
    )

    assert collector.consume_calls == 2
    assert collector.finalize_calls == 1
    assert collector.output_ids == model.case_output_ids
    assert model.encode_calls == 2
    assert model.decode_calls == 2
    assert model.forward_calls == 0
    assert set(result["case_exports"]) == {"failure_cases.jsonl", "evidence_cases.jsonl"}
    for name, rows in result["case_exports"].items():
        assert rows
        artifact_module._validate_epoch_jsonl(name, rows, epoch=0)


@pytest.mark.parametrize(
    ("case_collector", "case_export_provenance"),
    [(_ExplodingProtocolCollector(), None), (None, _p18_provenance(sample_count=8))],
)
def test_p19_case_export_arguments_must_be_paired_before_dino(
    case_collector: object,
    case_export_provenance: object,
) -> None:
    model = _EvalModel()
    model.train()
    with pytest.raises(ValueError, match="case_collector|case_export_provenance|together"):
        _evaluate(
            model=model,
            case_collector=case_collector,
            case_export_provenance=case_export_provenance,
        )
    assert model.encode_calls == 0
    assert model.training is True


def test_p19_rejects_non_protocol_case_collector_before_dino() -> None:
    model = _EvalModel()
    with pytest.raises(TypeError, match="RAELCaseExportCollector|consume|finalize"):
        _evaluate(
            model=model,
            case_collector=object(),
            case_export_provenance=_p18_provenance(sample_count=8),
        )
    assert model.encode_calls == 0


def test_p19_restores_model_mode_when_case_collector_fails() -> None:
    model = _EvalModel()
    model.train(False)
    with pytest.raises(RuntimeError, match="injected collector failure"):
        _evaluate(
            model=model,
            case_collector=_ExplodingProtocolCollector(),
            case_export_provenance=_p18_provenance(sample_count=8),
        )
    assert model.training is False


@pytest.mark.parametrize("insertion_order", ("full_first", "reversed", "randomized"))
def test_p19_accepts_unordered_branch_mapping_but_emits_model_protocol_order(
    insertion_order: str,
) -> None:
    result = _evaluate(model=_EvalModel(branch_insertion_order=insertion_order))
    assert [branch["name"] for branch in result["branch_metrics"]["branches"]] == list(
        EXPECTED_BRANCH_NAMES
    )


def test_p19_streams_one_shot_batches_and_matches_incremental_p12_hash() -> None:
    module = _module()
    stream = _OneShotTrackedBatches()
    expected_hash = _p12_split_hash(
        tuple(
            f"E:/bdd-oia/test/test-{index:04d}.jpg"
            for index in range(8)
        )
    )
    hasher = module._P12CompactSplitHasher()
    for index in range(8):
        hasher.add(f"E:/bdd-oia/test/test-{index:04d}.jpg")
    assert hasher.hexdigest() == expected_hash

    model = _EvalModel()
    result = _evaluate(
        model=model,
        batches=stream,  # type: ignore[arg-type]
        expected_test_split_hash=expected_hash,
    )
    assert model.encode_calls == 2
    assert stream.iterations == 1
    assert stream.released_batches == 2
    assert len(result["file_names"]) == 8


@pytest.mark.parametrize("dino_count", [None, 0, 2, torch.tensor(0), torch.tensor(2)])
def test_p19_rejects_missing_or_nonunit_dino_diagnostic(dino_count: int | Tensor | None) -> None:
    with pytest.raises(ValueError, match="dino_call_count"):
        _evaluate(model=_EvalModel(dino_call_count=dino_count))


def test_p19_rejects_missing_branch_or_nonidentical_full_branch_and_restores_mode() -> None:
    model = _EvalModel(omit_branch="evidence_shuffled")
    model.train(False)
    with pytest.raises(ValueError, match="14|branch"):
        _evaluate(model=model)
    assert model.training is False

    model = _EvalModel(mismatch_full=True)
    model.train()
    with pytest.raises(ValueError, match="full|final"):
        _evaluate(model=model)
    assert model.training is True

    model = _EvalModel(decode_raises=True)
    model.train(False)
    with pytest.raises(RuntimeError, match="injected"):
        _evaluate(model=model)
    assert model.training is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda batches: batches.__setitem__(0, {**batches[0], "split": "val"}),
        lambda batches: batches.__setitem__(0, {**batches[0], "file_names": ("",) * 4}),
        lambda batches: batches.__setitem__(1, {**batches[1], "file_names": batches[0]["file_names"]}),
        lambda batches: batches[0]["action_targets"].__setitem__((0, 0), 0.25),
        lambda batches: batches[0]["reason_targets"].__setitem__((0, 0), float("nan")),
        lambda batches: batches[0]["images"].__setitem__((0, 0), float("nan")),
    ],
)
def test_p19_rejects_non_test_or_invalid_input_contract(mutator: Any) -> None:
    batches = [_batch(0), _batch(4)]
    mutator(batches)
    model = _EvalModel()
    model.train()
    with pytest.raises((TypeError, ValueError), match="test|file|binary|finite|target"):
        _evaluate(
            model=model,
            batches=batches,
            expected_test_split_hash="e" * 64,
        )
    assert model.training is True


@pytest.mark.parametrize("field,value", [("fit_split", "test"), ("fit_split", "val")])
def test_p19_rejects_non_train_calib_calibration(field: str, value: object) -> None:
    action = copy.deepcopy(_calibration(4))
    action[field] = value
    with pytest.raises(ValueError, match="train_calib|calibration|digest"):
        _evaluate(action_calibration=action)


def test_p19_rejects_tampered_calibration_and_undefined_artifact_ranking() -> None:
    reason = copy.deepcopy(_calibration(21))
    reason["chosen"]["threshold"][0] = 123.0
    model = _EvalModel()
    with pytest.raises(ValueError, match="calibration|digest|threshold"):
        _evaluate(model=model, reason_calibration=reason)
    assert model.encode_calls == 0

    batches = [_batch(0), _batch(4)]
    for batch in batches:
        batch["action_targets"][:, 0] = 0.0
    with pytest.raises(ValueError, match="undefined|AP|AUC|ranking"):
        _evaluate(batches=batches)


def test_p19_binary_auc_handles_ties_and_keeps_undefined_nan_as_a_low_level_primitive() -> None:
    module = _module()
    assert module.binary_roc_auc(
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([0.0, 1.0, 0.0, 1.0]),
    ) == pytest.approx(0.5)
    assert module.binary_roc_auc(
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.0, 1.0]),
    ) == pytest.approx(1.0)
    assert math.isnan(module.binary_roc_auc(torch.tensor([0.0, 1.0]), torch.tensor([1.0, 1.0])))


def test_p19_rejects_malformed_action_schema_before_encode(tmp_path: Path) -> None:
    action_schema_path, reason_schema_path = _schema_paths()
    payload = yaml.safe_load(action_schema_path.read_text(encoding="utf-8"))
    payload["actions"][3]["id"] = 2
    tampered = tmp_path / "rael_action_semantics.yaml"
    tampered.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    model = _EvalModel()
    with pytest.raises((TypeError, ValueError), match="action|schema|id|name"):
        _evaluate(
            model=model,
            action_schema_path=tampered,
            reason_schema_path=reason_schema_path,
        )
    assert model.encode_calls == 0


def test_p19_rejects_calibration_source_or_expected_split_hash_mismatch_before_encode() -> None:
    test_like_ids = tuple(
        f"E:/bdd-oia/test/relabelled-calib-{index:04d}.jpg" for index in range(24)
    )
    cases = (
        {
            "action_calibration": _calibration(4, stable_ids=test_like_ids),
            "reason_calibration": _calibration(21),
        },
        {
            "action_calibration": _calibration(4),
            "reason_calibration": _calibration(21, stable_ids=_calib_ids(offset=100)),
        },
        {"expected_train_calib_split_hash": "f" * 64},
        {"expected_test_split_hash": "F" * 64},
    )
    for kwargs in cases:
        model = _EvalModel()
        with pytest.raises(ValueError, match="split|source|hash|lowercase|calib"):
            _evaluate(model=model, **kwargs)
        assert model.encode_calls == 0

    model = _EvalModel()
    with pytest.raises(ValueError, match="test|split|hash"):
        _evaluate(model=model, expected_test_split_hash="e" * 64)
    assert model.encode_calls == 2


def test_p19_tie_grouped_average_precision_is_row_order_stable() -> None:
    module = _module()
    scores = torch.tensor([0.8, 0.8, 0.3, 0.3], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
    permuted = torch.tensor([1, 0, 3, 2])
    expected = module.binary_average_precision_tie_stable(scores, targets)
    actual = module.binary_average_precision_tie_stable(scores[permuted], targets[permuted])
    assert expected == pytest.approx(actual)
    assert math.isfinite(expected)
