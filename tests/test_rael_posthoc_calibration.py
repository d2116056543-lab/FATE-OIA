"""P15 RED-to-GREEN contracts for frozen posthoc calibration."""

from __future__ import annotations

import copy
import importlib
import inspect
import json

import pytest
import torch


def _module():
    return importlib.import_module("fate_oia.utils.rael_posthoc_calibration")


def _sample(batch: int = 24, targets: int = 4) -> tuple[torch.Tensor, torch.Tensor, list[str], list[int]]:
    torch.manual_seed(1515 + batch + targets)
    logits = torch.randn(batch, targets, dtype=torch.float32)
    labels = (torch.sigmoid(logits + torch.linspace(-0.25, 0.25, targets)) > 0.55).to(torch.float32)
    if batch >= 8:
        labels[:4, -1] = 1.0
        labels[4:, -1] = 0.0
    ids = [f"calib/sequence-{index:04d}.jpg" for index in range(batch)]
    groups = [index % 2 for index in range(targets)]
    return logits, labels, ids, groups


def _fit(module, logits: torch.Tensor, labels: torch.Tensor, ids: list[str], groups: list[int]) -> dict:
    return module.fit_posthoc_calibration(
        raw_logits=logits,
        labels=labels,
        split="train_calib",
        stable_ids=ids,
        group_ids=groups,
    )


def test_fit_is_train_calib_only_cpu_float32_and_requires_detached_binary_inputs() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    for split in ("test", "val", "train"):
        with pytest.raises(ValueError, match="train_calib"):
            module.fit_posthoc_calibration(
                raw_logits=logits,
                labels=labels,
                split=split,
                stable_ids=ids,
                group_ids=groups,
            )
    with pytest.raises(ValueError, match="detached"):
        _fit(module, logits.requires_grad_(), labels, ids, groups)
    logits = logits.detach()
    with pytest.raises(ValueError, match="binary"):
        _fit(module, logits, labels.mul(0.5), ids, groups)
    with pytest.raises(ValueError, match="CPU float32"):
        _fit(module, logits.to(torch.float64), labels, ids, groups)
    with pytest.raises(ValueError, match="stable_ids or split_hash"):
        module.fit_posthoc_calibration(raw_logits=logits, labels=labels, split="train_calib", group_ids=groups)


def test_fit_records_exactly_four_real_candidate_families_and_deterministic_ties() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    first = _fit(module, logits, labels, ids, groups)
    second = _fit(module, logits.clone(), labels.clone(), list(ids), list(groups))
    assert [candidate["kind"] for candidate in first["candidates"]] == [
        "global_threshold",
        "group_threshold",
        "shrinkage_per_label_threshold",
        "positive_temperature_threshold",
    ]
    assert first == second
    assert first["fit_split"] == "train_calib"
    assert first["source"]["split_hash"]
    assert all(candidate["search"]["executed"] is True for candidate in first["candidates"])
    assert all(len(candidate["threshold"]) == 4 and len(candidate["temperature"]) == 4 for candidate in first["candidates"])
    assert all(float(value) > 0.0 for candidate in first["candidates"] for value in candidate["temperature"])
    assert first["chosen"]["kind"] in {candidate["kind"] for candidate in first["candidates"]} | {"identity"}


def test_shrinkage_uses_positive_support_counts_and_moves_low_support_threshold_toward_group() -> None:
    module = _module()
    logits, labels, ids, _ = _sample(batch=36, targets=4)
    labels[:, 0] = (torch.arange(36) % 2 == 0).to(torch.float32)
    labels[:, 1] = 0.0
    labels[0, 1] = 1.0
    labels[:, 2] = (torch.arange(36) % 3 == 0).to(torch.float32)
    labels[:, 3] = (torch.arange(36) % 4 == 0).to(torch.float32)
    result = _fit(module, logits, labels, ids, [0, 0, 1, 1])
    shrinkage = next(candidate for candidate in result["candidates"] if candidate["kind"] == "shrinkage_per_label_threshold")
    record = shrinkage["shrinkage"]
    assert record["support_counts"][1] == 1
    assert record["strength"] > 0.0
    group_threshold = record["group_thresholds"]["int:0"]
    label_threshold = record["label_thresholds_before_shrinkage"][1]
    deployed_threshold = shrinkage["threshold"][1]
    assert abs(deployed_threshold - group_threshold) < abs(label_threshold - group_threshold)


def test_apply_separates_raw_ranking_decision_and_float64_diagnostic_margin() -> None:
    module = _module()
    logits = torch.tensor(
        [
            [1.0000000, -0.1000000, 0.0, 0.0000001],
            [1.0000001, -0.0999999, 0.0, 0.0000002],
            [0.9999999, -0.1000001, 0.0, -0.0000001],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    ids = ["nearby-0.jpg", "nearby-1.jpg", "nearby-2.jpg"]
    groups = [0, 0, 1, 1]
    result = _fit(module, logits, labels, ids, groups)
    snapshot = copy.deepcopy(result)
    output = module.apply_posthoc_calibration(logits.clone().requires_grad_(), result)
    assert set(output) == {"ranking_logits", "decision", "diagnostic_margin", "ranking_source"}
    assert output["ranking_source"] == "raw_logits"
    assert output["ranking_logits"].dtype == torch.float32
    assert output["ranking_logits"].requires_grad is False
    assert output["ranking_logits"].data_ptr() != logits.data_ptr()
    assert torch.equal(output["ranking_logits"], logits)
    assert output["decision"].dtype == torch.bool
    assert output["diagnostic_margin"].dtype == torch.float64
    threshold = torch.tensor(result["chosen"]["threshold"], dtype=torch.float64)
    temperature = torch.tensor(result["chosen"]["temperature"], dtype=torch.float64)
    assert torch.equal(output["decision"], logits.to(torch.float64) > temperature * threshold)
    assert torch.equal(output["diagnostic_margin"], logits.to(torch.float64) - temperature * threshold)
    assert result == snapshot
    assert module.multi_label_metrics(output["ranking_logits"], labels)["map"] == pytest.approx(
        module.multi_label_metrics(logits, labels)["map"], abs=1.0e-7
    )
    assert not any(isinstance(value, torch.Tensor) for value in result.values())


def test_threshold_rms_strict_bound_and_safety_fallback_never_returns_illegal_candidate() -> None:
    module = _module()
    assert module._threshold_rms_is_compliant(torch.tensor([0.34]), raw_logit_rms=1.0, fraction=0.35)
    assert not module._threshold_rms_is_compliant(torch.tensor([0.35]), raw_logit_rms=1.0, fraction=0.35)
    candidates = [
        {"kind": "global_threshold", "eligible": True, "threshold": [0.1, 0.1], "temperature": [1.0, 1.0], "metrics": {"mf1": 0.40}},
        {"kind": "group_threshold", "eligible": True, "threshold": [0.1, 0.1], "temperature": [1.0, 1.0], "metrics": {"mf1": 0.50}},
        {"kind": "shrinkage_per_label_threshold", "eligible": False, "threshold": [0.1, 0.1], "temperature": [1.0, 1.0], "metrics": {"mf1": 0.50}},
        {"kind": "positive_temperature_threshold", "eligible": False, "threshold": [0.1, 0.1], "temperature": [1.0, 1.0], "metrics": {"mf1": 0.50}},
    ]
    chosen, fallback = module._select_safe_candidate(candidates=candidates, raw_fixed_mf1=0.60, targets=2)
    assert chosen["kind"] == "identity" and fallback["used"] is True
    assert "global" in fallback["reason"]


def test_fit_preserves_raw_fixed_when_calibration_candidates_are_illegal_or_not_improving() -> None:
    module = _module()
    logits = torch.zeros(12, 4, dtype=torch.float32)
    labels = torch.zeros(12, 4, dtype=torch.float32)
    labels[:3, 0] = 1.0
    result = _fit(module, logits, labels, [f"constant-{index}.jpg" for index in range(12)], [0, 0, 1, 1])
    assert result["chosen"]["kind"] == "identity"
    assert result["fallback"]["used"] is True
    assert result["chosen"]["threshold"] == [0.0] * 4
    assert result["chosen"]["temperature"] == [1.0] * 4


@pytest.mark.parametrize("targets", [4, 21])
def test_zero_logits_identity_uses_the_same_strict_raw_fixed_decision_rule(targets: int) -> None:
    module = _module()
    logits = torch.zeros(8, targets, dtype=torch.float32)
    labels = torch.zeros_like(logits)
    labels[:2, 0] = 1.0
    result = _fit(module, logits, labels, [f"zero-{index}.jpg" for index in range(8)], [index % 2 for index in range(targets)])
    output = module.apply_posthoc_calibration(logits, result)
    expected = logits > 0.0
    assert result["chosen"]["kind"] == "identity"
    assert torch.equal(output["decision"], expected)
    strict_metrics = module._decision_ranking_metrics(logits, labels, expected)
    assert result["train_calib_metrics"]["raw_fixed"]["mf1"] == pytest.approx(strict_metrics["mf1"])
    assert result["train_calib_metrics"]["deploy"]["mf1"] == pytest.approx(strict_metrics["mf1"])


def test_hierarchical_selection_has_primary_global_fallback_and_identity_paths() -> None:
    module = _module()
    def candidate(kind: str, mf1: float, eligible: bool = True) -> dict:
        return {
            "kind": kind,
            "eligible": eligible,
            "threshold": [0.1, 0.1],
            "temperature": [1.0, 1.0],
            "metrics": {"mf1": mf1},
        }

    primary, fallback = module._select_safe_candidate(
        candidates=[
            candidate("global_threshold", 0.99),
            candidate("group_threshold", 0.70),
            candidate("shrinkage_per_label_threshold", 0.69),
            candidate("positive_temperature_threshold", 0.68),
        ],
        raw_fixed_mf1=0.60,
        targets=2,
    )
    assert primary["kind"] == "group_threshold"
    assert fallback == {"used": False, "reason": "primary_candidate_passed_guard", "path": "primary"}

    global_choice, global_fallback = module._select_safe_candidate(
        candidates=[
            candidate("global_threshold", 0.60),
            candidate("group_threshold", 0.50),
            candidate("shrinkage_per_label_threshold", 0.51),
            candidate("positive_temperature_threshold", 0.49),
        ],
        raw_fixed_mf1=0.60,
        targets=2,
    )
    assert global_choice["kind"] == "global_threshold"
    assert global_fallback == {"used": True, "reason": "primary_candidate_failed_guard", "path": "global_fallback"}

    identity, identity_fallback = module._select_safe_candidate(
        candidates=[
            candidate("global_threshold", 0.60, eligible=False),
            candidate("group_threshold", 0.50),
            candidate("shrinkage_per_label_threshold", 0.51),
            candidate("positive_temperature_threshold", 0.49),
        ],
        raw_fixed_mf1=0.60,
        targets=2,
    )
    assert identity["kind"] == "identity"
    assert identity_fallback["used"] is True and identity_fallback["path"] == "identity"


def test_serialization_digest_rejects_inplace_tampering_and_test_oracle_is_diagnostic_only() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    result = _fit(module, logits, labels, ids, groups)
    assert isinstance(result["payload_sha256"], str) and len(result["payload_sha256"]) == 64
    payload = module.serialize_calibration_result(result)
    restored = module.deserialize_calibration_result(payload)
    assert restored == result
    before = copy.deepcopy(restored)
    _ = module.apply_posthoc_calibration(logits, restored)
    assert restored == before
    for mutate in (
        lambda value: value["chosen"]["threshold"].__setitem__(0, value["chosen"]["threshold"][0] + 0.1),
        lambda value: value["chosen"].__setitem__("raw_logit_rms", value["chosen"]["raw_logit_rms"] + 1.0),
        lambda value: value["chosen"].__setitem__("kind", "global_threshold"),
        lambda value: value["chosen"]["temperature"].__setitem__(0, 0.0),
        lambda value: value["candidates"][0]["metrics"].__setitem__("mf1", 0.999),
    ):
        mutated = copy.deepcopy(result)
        mutate(mutated)
        with pytest.raises(ValueError, match="digest"):
            module.apply_posthoc_calibration(logits, mutated)
        with pytest.raises(ValueError, match="digest"):
            module.serialize_calibration_result(mutated)
    tampered_payload = json.loads(payload)
    tampered_payload["chosen"]["threshold"][0] += 0.1
    with pytest.raises(ValueError, match="digest"):
        module.deserialize_calibration_result(json.dumps(tampered_payload, sort_keys=True))
    semantic_tamper = copy.deepcopy(result)
    semantic_tamper["chosen"]["raw_logit_rms"] += 1.0
    with pytest.raises(ValueError, match="chosen calibration"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(semantic_tamper))
    semantic_tamper = copy.deepcopy(result)
    semantic_tamper["chosen"]["temperature"][0] = 0.0
    with pytest.raises(ValueError, match="chosen calibration"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(semantic_tamper))
    eligible_alternative = next(
        candidate
        for candidate in result["candidates"]
        if candidate["eligible"] and candidate["kind"] != result["chosen"]["kind"]
    )
    semantic_tamper = copy.deepcopy(result)
    semantic_tamper["chosen"] = copy.deepcopy(eligible_alternative)
    semantic_tamper["fallback"] = {"used": False, "reason": "primary_candidate_passed_guard", "path": "primary"}
    with pytest.raises(ValueError, match="chosen metrics"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(semantic_tamper))
    semantic_tamper = copy.deepcopy(result)
    semantic_tamper["fallback"]["reason"] = "forged_fallback_reason"
    with pytest.raises(ValueError, match="hierarchical"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(semantic_tamper))
    diagnostic = module.diagnostic_test_oracle(raw_logits=logits, labels=labels)
    assert diagnostic["kind"] == "test_oracle_diagnostic"
    assert "chosen" not in diagnostic and "threshold" not in diagnostic
    signature = inspect.signature(module.fit_posthoc_calibration)
    assert not {"test_logits", "test_labels", "test_metrics", "val_logits", "val_labels"}.intersection(signature.parameters)


@pytest.mark.parametrize("targets", [4, 21])
def test_fit_handles_ties_empty_labels_batch_one_and_k4_k21(targets: int) -> None:
    module = _module()
    logits = torch.zeros(1 if targets == 4 else 24, targets, dtype=torch.float32)
    labels = torch.zeros_like(logits)
    if logits.shape[0] > 1:
        labels[0, 0] = 1.0
        labels[1::3, min(1, targets - 1)] = 1.0
    ids = [f"tie-{index}.jpg" for index in range(logits.shape[0])]
    result = _fit(module, logits, labels, ids, [index % 3 for index in range(targets)])
    assert result["targets"] == targets
    assert len(result["chosen"]["threshold"]) == targets
    deployed = module.apply_posthoc_calibration(logits, result)
    assert torch.isfinite(deployed["ranking_logits"]).all()
    assert torch.isfinite(deployed["diagnostic_margin"]).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="P15 CUDA BF16 apply requires CUDA")
def test_apply_supports_cuda_bf16_without_parameter_gradients() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    result = _fit(module, logits, labels, ids, groups)
    gpu_logits = logits.cuda().to(torch.bfloat16).requires_grad_()
    output = module.apply_posthoc_calibration(gpu_logits, result)
    assert output["ranking_logits"].device.type == "cuda"
    assert output["ranking_logits"].dtype == torch.bfloat16
    assert output["ranking_logits"].requires_grad is False
    assert output["diagnostic_margin"].dtype == torch.float64
    assert torch.isfinite(output["ranking_logits"]).all()
    assert torch.isfinite(output["diagnostic_margin"]).all()


def test_semantic_integrity_rejects_rehashed_non_grid_temperature_and_out_of_range_metrics() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    result = _fit(module, logits, labels, ids, groups)
    assert result["allowed_temperature_grid"] == list(module.TEMPERATURE_GRID)
    assert all(
        value in result["allowed_temperature_grid"]
        for candidate in result["candidates"]
        for value in candidate["temperature"]
    )

    for mutate, expected in (
        (
            lambda value: value["candidates"][0]["temperature"].__setitem__(0, 1.0e30),
            "temperature",
        ),
        (
            lambda value: value["candidates"][0]["metrics"].__setitem__("mf1", 5.0),
            "metrics",
        ),
        (
            lambda value: value["candidates"].append(copy.deepcopy(value["candidates"][0])),
            "candidate",
        ),
        (
            lambda value: value["candidates"][0].__setitem__("schema_version", "forged"),
            "candidate",
        ),
    ):
        tampered = copy.deepcopy(result)
        mutate(tampered)
        with pytest.raises(ValueError, match=expected):
            module.apply_posthoc_calibration(logits, module._with_payload_digest(tampered))


def test_fit_apply_and_payload_fail_closed_for_nonfinite_values() -> None:
    module = _module()
    logits, labels, ids, groups = _sample()
    for value in (float("nan"), float("inf"), float("-inf")):
        invalid_logits = logits.clone()
        invalid_logits[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            _fit(module, invalid_logits, labels, ids, groups)
        invalid_labels = labels.clone()
        invalid_labels[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            _fit(module, logits, invalid_labels, ids, groups)

    result = _fit(module, logits, labels, ids, groups)
    invalid_apply = logits.clone()
    invalid_apply[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.apply_posthoc_calibration(invalid_apply, result)

    tampered = copy.deepcopy(result)
    tampered["candidates"][0]["threshold"][0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(tampered))


def test_source_descriptor_reuses_p12_canonical_ids_and_rejects_collisions_or_unsafe_paths() -> None:
    module = _module()
    logits, labels, _, groups = _sample(batch=8)
    first_ids = [f"C:\\BDD\\Sequence\\Frame-{index}.JPG" for index in range(8)]
    second_ids = [f"c:/bdd/sequence/frame-{index}.jpg" for index in range(8)]
    first = _fit(module, logits, labels, first_ids, groups)
    second = _fit(module, logits, labels, second_ids, groups)
    assert first["source"]["split_hash"] == second["source"]["split_hash"]
    assert first["source"]["canonicalization"] == "P12_canonicalize_sample_id"
    assert first["source"]["canonical_id_order"] == "input_row_order"

    duplicate = list(first_ids)
    duplicate[1] = "c:/bdd/sequence/frame-0.jpg"
    with pytest.raises(ValueError, match="unique"):
        _fit(module, logits, labels, duplicate, groups)
    unsafe = list(first_ids)
    unsafe[0] = "C:/BDD/../../escape.jpg"
    with pytest.raises(ValueError):
        _fit(module, logits, labels, unsafe, groups)


def test_group_ids_use_typed_namespace_without_int_string_collisions() -> None:
    module = _module()
    logits, labels, ids, _ = _sample(batch=24, targets=4)
    group_ids: list[int | str] = [1, "1", 2, "2"]
    first = _fit(module, logits, labels, ids, group_ids)
    second = _fit(module, logits, labels, list(ids), list(group_ids))
    assert first == second
    group_candidate = next(candidate for candidate in first["candidates"] if candidate["kind"] == "group_threshold")
    shrink_candidate = next(candidate for candidate in first["candidates"] if candidate["kind"] == "shrinkage_per_label_threshold")
    expected = ["int:1", "str:1", "int:2", "str:2"]
    assert group_candidate["search"]["group_ids"] == expected
    assert sorted(shrink_candidate["shrinkage"]["group_thresholds"]) == sorted(expected)
    assert len(shrink_candidate["shrinkage"]["group_thresholds"]) == 4
    for invalid_groups in ([True, 1, 2, 3], ["", 1, 2, 3], ["has space", 1, 2, 3]):
        with pytest.raises((TypeError, ValueError)):
            _fit(module, logits, labels, ids, invalid_groups)


def _refresh_candidate_rms_and_eligibility(module, candidate: dict) -> None:
    threshold = torch.tensor(candidate["threshold"], dtype=torch.float64)
    candidate["threshold_rms"] = float(threshold.square().mean().sqrt().item())
    candidate["eligible"] = module._threshold_rms_is_compliant(
        threshold,
        raw_logit_rms=float(candidate["raw_logit_rms"]),
    )


def test_rehashed_payload_still_rejects_semantically_inconsistent_metrics_thresholds_and_deploy() -> None:
    module = _module()
    logits, labels, ids, groups = _sample(batch=36)
    result = _fit(module, logits, labels, ids, groups)
    assert result["chosen"]["metrics"] == result["train_calib_metrics"]["deploy"]
    assert result["integrity"]["model"] == "internal_consistency+accidental_corruption"
    assert "not_adversarial_resigning" in result["integrity"]["sha256_limitation"]
    assert all("decision_stats" in candidate and "provenance" in candidate for candidate in result["candidates"])

    metric_tamper = copy.deepcopy(result)
    metric_tamper["candidates"][0]["metrics"]["per_label_f1"] = [0.99] * logits.shape[1]
    metric_tamper["candidates"][0]["metrics"]["mf1"] = 0.99
    with pytest.raises(ValueError, match="decision statistics"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(metric_tamper))

    shrink_tamper = copy.deepcopy(result)
    shrink = next(candidate for candidate in shrink_tamper["candidates"] if candidate["kind"] == "shrinkage_per_label_threshold")
    shrink["threshold"] = [float(value) * 0.5 + 0.123 for value in shrink["threshold"]]
    _refresh_candidate_rms_and_eligibility(module, shrink)
    with pytest.raises(ValueError, match="float32 formula"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(shrink_tamper))

    global_tamper = copy.deepcopy(result)
    global_candidate = global_tamper["candidates"][0]
    global_candidate["threshold"][0] += 0.25
    _refresh_candidate_rms_and_eligibility(module, global_candidate)
    with pytest.raises(ValueError, match="global candidate"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(global_tamper))

    deploy_tamper = copy.deepcopy(result)
    fake_deploy = copy.deepcopy(deploy_tamper["train_calib_metrics"]["deploy"])
    fake_deploy["per_label_f1"] = [0.0] * logits.shape[1]
    fake_deploy["mf1"] = 0.0
    deploy_tamper["train_calib_metrics"]["deploy"] = fake_deploy
    with pytest.raises(ValueError, match="chosen metrics"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(deploy_tamper))


def test_rehashed_support_tamper_cannot_escape_when_label_threshold_equals_group_threshold() -> None:
    module = _module()
    logits, labels, ids, groups = _sample(batch=72)
    result = _fit(module, logits, labels, ids, groups)
    tampered = copy.deepcopy(result)
    shrink = next(candidate for candidate in tampered["candidates"] if candidate["kind"] == "shrinkage_per_label_threshold")
    group_id = shrink["search"]["group_ids"][0]
    group_threshold = shrink["shrinkage"]["group_thresholds"][group_id]
    shrink["shrinkage"]["label_thresholds_before_shrinkage"][0] = group_threshold
    shrink["threshold"][0] = group_threshold
    shrink["shrinkage"]["support_counts"][0] += 7
    _refresh_candidate_rms_and_eligibility(module, shrink)
    with pytest.raises(ValueError, match="decision-statistics support"):
        module.apply_posthoc_calibration(logits, module._with_payload_digest(tampered))


@pytest.mark.parametrize(
    ("batch", "seed"),
    [(512, 31), (512, 47), (4096, 31), (4096, 47), (16082, 31), (16082, 47)],
)
def test_shared_float32_shrinkage_recomputation_accepts_legal_multi_scale_multi_seed_fits(batch: int, seed: int) -> None:
    module = _module()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(batch, 21, generator=generator, dtype=torch.float32)
    labels = (torch.rand(batch, 21, generator=generator) > 0.81).to(torch.float32)
    result = _fit(
        module,
        logits,
        labels,
        [f"C:/BDD-OIA/train/seed-{seed}/frame-{index:05d}.jpg" for index in range(batch)],
        [index % 4 for index in range(21)],
    )
    shrink = next(candidate for candidate in result["candidates"] if candidate["kind"] == "shrinkage_per_label_threshold")
    assert shrink["shrinkage"]["support_counts"] == shrink["decision_stats"]["support"]
    restored = module.deserialize_calibration_result(module.serialize_calibration_result(result))
    assert restored == result
