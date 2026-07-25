"""P4 behavioral contracts for compositional RAEL semantic reason queries."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs" / "rael_reason_semantics.yaml"


def _module() -> Any:
    spec = importlib.util.find_spec("fate_oia.models.rael_semantic_reason")
    assert spec is not None, "P4 semantic reason module must exist before its contract can run"
    return importlib.import_module("fate_oia.models.rael_semantic_reason")


class _FakeFieldReader:
    """P3-compatible test double whose readout is deliberately deterministic."""

    def __init__(self, dim: int = 384, output_dtype: torch.dtype | None = None) -> None:
        self.dim = dim
        self.output_dtype = output_dtype
        self.calls = 0

    def read(self, prepared: dict[str, Tensor], queries: Tensor, group_name: str | None = None) -> dict[str, Tensor | str | None]:
        self.calls += 1
        batch, count, dim = queries.shape
        assert dim == self.dim
        field_offset = prepared["field_offset"].to(device=queries.device, dtype=queries.dtype)
        readout = queries + field_offset.view(batch, 1, 1)
        if self.output_dtype is not None:
            readout = readout.to(self.output_dtype)
        return {
            "group_name": group_name,
            "readout": readout,
            "layer_weights": torch.full((batch, count, 4), 0.25, device=queries.device, dtype=queries.dtype),
        }


def _evidence(module: Any, *, batch: int = 2, count: int = 7, value: float = 0.0) -> Any:
    return module.EvidenceReadBundle(
        tokens=torch.full((batch, count, 384), value, dtype=torch.float32),
        valid_mask=torch.ones((batch, count), dtype=torch.bool),
    )


def _schema_payload() -> dict[str, Any]:
    payload = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_schema(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_p4_component_omission_is_a_normal_assertion_not_collection_error() -> None:
    _module()


def test_schema_loader_strictly_rejects_duplicate_keys_and_missing_components(tmp_path: Path) -> None:
    module = _module()
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    with pytest.raises(Exception, match="duplicate"):
        module.load_reason_semantic_schema(duplicate)

    omitted = tmp_path / "omitted.yaml"
    payload = SCHEMA_PATH.read_text(encoding="utf-8").replace("entity: traffic_control, ", "", 1)
    omitted.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        module.load_reason_semantic_schema(omitted)


def test_schema_rows_are_typed_immutable_soft_priors_with_valid_mirrors() -> None:
    module = _module()
    rows = module.load_reason_semantic_schema(SCHEMA_PATH)
    assert len(rows) == 21
    assert tuple(row.id for row in rows) == tuple(range(21))
    assert all(getattr(row, "__dataclass_params__").frozen for row in rows)
    assert set(row.role for row in rows) == module.ROLE_NAMES
    with pytest.raises((AttributeError, TypeError)):
        rows[0].entity = "mutated"


def test_schema_rejects_hard_compatibility_and_invalid_mirror(tmp_path: Path) -> None:
    module = _module()
    hard = tmp_path / "hard.yaml"
    hard.write_text(SCHEMA_PATH.read_text(encoding="utf-8").replace("hard_action_masks: false", "hard_action_masks: true"), encoding="utf-8")
    with pytest.raises(ValueError, match="soft priors"):
        module.load_reason_semantic_schema(hard)

    mirror = tmp_path / "mirror.yaml"
    mirror.write_text(SCHEMA_PATH.read_text(encoding="utf-8").replace("mirror_partner: 15", "mirror_partner: 16", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="mirror"):
        module.load_reason_semantic_schema(mirror)


def test_schema_rejects_every_extra_row_key_including_compatibility_aliases(tmp_path: Path) -> None:
    module = _module()
    payload = _schema_payload()
    payload["reasons"][0]["compatibility_alias"] = "forward"
    candidate = tmp_path / "extra_alias.yaml"
    _write_schema(candidate, payload)
    with pytest.raises(ValueError, match="exactly the nine"):
        module.load_reason_semantic_schema(candidate)

    non_compatibility = _schema_payload()
    non_compatibility["reasons"][1]["unexpected_metadata"] = "must not silently pass"
    candidate = tmp_path / "extra_metadata.yaml"
    _write_schema(candidate, non_compatibility)
    with pytest.raises(ValueError, match="exactly the nine"):
        module.load_reason_semantic_schema(candidate)


def test_schema_rejects_unknown_or_misspelled_top_level_keys(tmp_path: Path) -> None:
    module = _module()
    unknown = _schema_payload()
    unknown["unexpected_top_level"] = True
    candidate = tmp_path / "unknown_top_level.yaml"
    _write_schema(candidate, unknown)
    with pytest.raises(ValueError, match="exactly the four top-level"):
        module.load_reason_semantic_schema(candidate)

    misspelled = _schema_payload()
    misspelled["schema_versoin"] = misspelled.pop("schema_version")
    candidate = tmp_path / "misspelled_top_level.yaml"
    _write_schema(candidate, misspelled)
    with pytest.raises(ValueError, match="exactly the four top-level"):
        module.load_reason_semantic_schema(candidate)


def test_schema_rejects_semantically_invalid_mirror_doubles(tmp_path: Path) -> None:
    module = _module()

    state_mismatch = _schema_payload()
    state_mismatch["reasons"][15]["state"] = "obstructing"
    candidate = tmp_path / "state_mismatch.yaml"
    _write_schema(candidate, state_mismatch)
    with pytest.raises(ValueError, match="state"):
        module.load_reason_semantic_schema(candidate)

    wrong_pair = _schema_payload()
    rows = {row["id"]: row for row in wrong_pair["reasons"]}
    rows[9]["mirror_partner"], rows[16]["mirror_partner"] = 16, 9
    rows[10]["mirror_partner"], rows[15]["mirror_partner"] = 15, 10
    candidate = tmp_path / "wrong_pair.yaml"
    _write_schema(candidate, wrong_pair)
    with pytest.raises(ValueError, match="mirror"):
        module.load_reason_semantic_schema(candidate)

    non_directional_role = _schema_payload()
    non_directional_role["reasons"][9]["role"] = "forward_support"
    candidate = tmp_path / "non_directional_role.yaml"
    _write_schema(candidate, non_directional_role)
    with pytest.raises(ValueError, match="left/right role"):
        module.load_reason_semantic_schema(candidate)


def test_schema_forces_directional_name_state_and_evidence_tokens_to_mirror(tmp_path: Path) -> None:
    module = _module()

    repeated_name = _schema_payload()
    repeated_name["reasons"][15]["name"] = repeated_name["reasons"][9]["name"]
    candidate = tmp_path / "left_name_to_left_name.yaml"
    _write_schema(candidate, repeated_name)
    with pytest.raises(ValueError, match="name"):
        module.load_reason_semantic_schema(candidate)

    repeated_state = _schema_payload()
    repeated_state["reasons"][19]["state"] = "left_allowed"
    candidate = tmp_path / "left_state_to_left_state.yaml"
    _write_schema(candidate, repeated_state)
    with pytest.raises(ValueError, match="state"):
        module.load_reason_semantic_schema(candidate)

    repeated_evidence = _schema_payload()
    repeated_evidence["reasons"][9]["explicit_evidence_families"][0] = "left_boundary"
    repeated_evidence["reasons"][15]["explicit_evidence_families"][0] = "left_boundary"
    candidate = tmp_path / "left_evidence_to_left_evidence.yaml"
    _write_schema(candidate, repeated_evidence)
    with pytest.raises(ValueError, match="explicit evidence families"):
        module.load_reason_semantic_schema(candidate)


def test_compositional_query_is_bounded_trainable_and_has_no_hard_action_mask() -> None:
    module = _module()
    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    query = semantic.compositional_queries(batch_size=2)
    assert query.shape == (2, 21, 384)
    assert semantic.reason_residual.requires_grad
    base = semantic.query_norm(
        semantic.entity_embedding(semantic.entity_ids)
        + semantic.state_embedding(semantic.state_ids)
        + semantic.sector_embedding(semantic.sector_ids)
        + semantic.role_embedding(semantic.role_ids)
    )
    with torch.no_grad():
        semantic.reason_residual.fill_(1.0e6)
    bounded = semantic.compositional_queries(batch_size=1)[0]
    assert float((bounded - base).abs().max()) <= 0.100001
    assert not hasattr(semantic, "action_compatibility_mask")
    assert semantic.parameter_owner == "semantic_reason"


def test_read_uses_both_p3_field_and_future_ledger_evidence() -> None:
    module = _module()
    torch.manual_seed(7)
    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    field = _FakeFieldReader()
    prepared = {"field_offset": torch.tensor([0.0, 0.5])}
    evidence_a = _evidence(module, value=0.0)
    evidence_b = _evidence(module, value=3.0)
    out_a = semantic(field, prepared, evidence_a)
    out_b = semantic(field, prepared, evidence_b)
    assert field.calls == 2
    assert out_a["semantic_reason_tokens"].shape == (2, 21, 384)
    assert out_a["layer_weights"].shape == (2, 21, 4)
    assert out_a["evidence_weights"].shape == (2, 21, 7)
    assert not torch.allclose(out_a["semantic_reason_tokens"], out_b["semantic_reason_tokens"])
    assert torch.allclose(out_a["evidence_weights"].sum(dim=-1), torch.ones(2, 21), atol=1e-6)


def test_semantic_reason_reads_the_real_p3_multilayer_field() -> None:
    module = _module()
    from fate_oia.models.rael_multilayer_field import RAELMultiLayerField

    torch.manual_seed(19)
    field = RAELMultiLayerField()
    field.eval()
    with torch.no_grad():
        prepared = field.precompute(
            torch.randn(1, 4, 3600, 384),
            torch.randn(1, 4, 384),
        )
    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    output = semantic(field, prepared, _evidence(module, batch=1, count=3, value=0.25))
    assert output["semantic_reason_tokens"].shape == (1, 21, 384)
    assert output["layer_weights"].shape == (1, 21, 4)
    assert output["evidence_weights"].shape == (1, 21, 3)


def test_semantic_reason_normalizes_a_bf16_p3_read_boundary() -> None:
    module = _module()
    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    field = _FakeFieldReader(output_dtype=torch.bfloat16)
    evidence = module.EvidenceReadBundle(
        tokens=torch.ones((2, 3, 384), dtype=torch.bfloat16),
        valid_mask=torch.ones((2, 3), dtype=torch.bool),
    )
    output = semantic(field, {"field_offset": torch.zeros(2)}, evidence)
    assert output["semantic_reason_tokens"].dtype == torch.float32
    assert torch.isfinite(output["semantic_reason_tokens"]).all()


def test_evidence_bundle_fails_fast_for_nonfinite_construction_and_forward_mutation() -> None:
    module = _module()
    with pytest.raises(ValueError, match="tokens must be finite"):
        module.EvidenceReadBundle(
            tokens=torch.full((1, 2, 384), float("nan")),
            valid_mask=torch.ones((1, 2), dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="valid_mask must be finite"):
        module.EvidenceReadBundle(
            tokens=torch.zeros((1, 2, 384)),
            valid_mask=torch.tensor([[float("inf"), 1.0]]),
        )

    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    field = _FakeFieldReader()
    evidence = _evidence(module, batch=1, count=2, value=0.0)
    with torch.no_grad():
        evidence.tokens.fill_(float("nan"))
    with pytest.raises(ValueError, match="tokens must be finite"):
        semantic(field, {"field_offset": torch.zeros(1)}, evidence)


def test_evidence_interface_rejects_field_only_and_label_leakage() -> None:
    module = _module()
    semantic = module.RAELSemanticReason(SCHEMA_PATH, dim=384)
    field = _FakeFieldReader()
    with pytest.raises(TypeError, match="EvidenceReadBundle"):
        semantic(field, {"field_offset": torch.zeros(2)}, None)
    parameter_names = set(inspect.signature(semantic.forward).parameters)
    forbidden = {"reason_labels", "reason_logits", "private_tokens", "pu_state", "bdd_geometry"}
    assert forbidden.isdisjoint(parameter_names)
    source = inspect.getsource(module).lower()
    assert all(token not in source for token in ("bert", "clip", "vlm", "hashing"))
