"""P7 behavioral contracts for the RAEL action-category foundation."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]


def _module() -> Any:
    spec = importlib.util.find_spec("fate_oia.models.rael_category_foundation")
    assert spec is not None, "P7 action-category module must exist before its contract can run"
    return importlib.import_module("fate_oia.models.rael_category_foundation")


class _FormalP3Reader:
    """A deterministic P3 protocol double; P3 remains the layer-mixing owner."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.calls: list[dict[str, Any]] = []

    def read(
        self,
        prepared: dict[str, Tensor],
        queries: Tensor,
        group_name: str | None = None,
    ) -> dict[str, Tensor | str | None]:
        self.calls.append(
            {"group_name": group_name, "prepared_id": id(prepared), "queries": queries.detach().clone()}
        )
        batch, count, dim = queries.shape
        assert dim == self.dim
        offset = prepared["field_offset"].to(device=queries.device, dtype=queries.dtype)
        assert offset.shape == (batch, dim)
        weights = torch.tensor([0.10, 0.20, 0.30, 0.40], device=queries.device, dtype=queries.dtype)
        return {
            "group_name": group_name,
            "readout": queries + offset.unsqueeze(1),
            "layer_weights": weights.view(1, 1, 4).expand(batch, count, -1),
        }


def _prepared(batch: int = 2, perturb: float = 0.0) -> dict[str, Tensor]:
    base = torch.linspace(-0.4, 0.4, 384, dtype=torch.float32).view(1, 384).expand(batch, -1)
    return {"field_offset": base + perturb}


def _uniformize_attention(model: Any) -> None:
    attention = model.action_self_attention
    dim = model.dim
    with torch.no_grad():
        attention.in_proj_weight.zero_()
        attention.in_proj_bias.zero_()
        attention.in_proj_weight[2 * dim : 3 * dim].copy_(torch.eye(dim))
        attention.out_proj.weight.copy_(torch.eye(dim))
        attention.out_proj.bias.zero_()


def test_p7_component_omission_is_a_normal_assertion_not_collection_error() -> None:
    _module()


def test_four_queries_share_side_base_and_have_exact_mirror_deltas() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()
    explicit_six = module.RAELActionCategoryFoundation(num_heads=6)
    assert model.num_heads == 6
    assert explicit_six.num_heads == 6
    queries = model.compositional_queries(batch_size=2)
    components = model.query_components()

    assert module.ACTION_NAMES == ("forward", "stop", "left", "right")
    assert queries.shape == (2, 4, 384)
    side = components["side_shared"].view(1, 1, -1)
    assert torch.allclose((queries[:, 2] + queries[:, 3]) * 0.5, side)
    assert torch.allclose(queries[:, 2] - side, -(queries[:, 3] - side))
    names = tuple(name for name, _ in model.named_parameters())
    assert not any("left_query" in name or "right_query" in name for name in names)
    assert model.parameter_owner == "action_category"
    assert model.learning_rate == pytest.approx(2.0e-4)


def test_reads_formal_p3_multilayer_api_once_with_complete_action_group() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()
    reader = _FormalP3Reader()
    prepared = _prepared()

    output = model(reader, prepared)

    assert len(reader.calls) == 1
    assert reader.calls[0]["group_name"] == "action_category"
    assert reader.calls[0]["prepared_id"] == id(prepared)
    assert reader.calls[0]["queries"].shape == (2, 4, 384)
    assert output["readouts"].shape == (2, 4, 384)
    assert output["layer_weights"].shape == (2, 4, 4)
    assert torch.allclose(output["layer_weights"].sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_one_action_self_attention_mixes_forward_perturbation_into_stop_token() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()
    _uniformize_attention(model)
    reader = _FormalP3Reader()

    before = model(reader, _prepared(batch=1))["action_visual_tokens"].detach()
    with torch.no_grad():
        model.forward_query.add_(torch.linspace(-1.0, 1.0, model.dim))
    after = model(reader, _prepared(batch=1))["action_visual_tokens"].detach()

    assert isinstance(model.action_self_attention, nn.MultiheadAttention)
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_project_global_is_action_specific_visual_only_and_never_emits_formal_bridge_key() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()
    assert model.global_head.weight.shape == (4, 384)
    assert model.global_head.bias.shape == (4,)
    reader = _FormalP3Reader()
    output = model(reader, _prepared(batch=1))

    visual_tokens = output["action_visual_tokens"]
    expected = torch.einsum("bad,ad->ba", visual_tokens, model.global_head.weight) + model.global_head.bias
    assert torch.allclose(model.project_global(visual_tokens), expected)
    assert torch.allclose(output["z_A_global_visual"], expected)
    assert "z_A_global" not in output
    assert output["z_A_global_visual"].shape == (1, 4)
    assert "action_tokens" not in output
    assert "global_logits" not in output


def test_project_global_rejects_wrong_rank_or_trailing_singleton_output() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()
    tokens = torch.randn(2, 4, 384)

    with pytest.raises(ValueError, match="\\[B,4,384\\]"):
        model.project_global(tokens[:, :3])

    class _WrongHead(nn.Module):
        def forward(self, values: Tensor) -> Tensor:
            return values[..., :1]

    model.global_head = _WrongHead()
    with pytest.raises(RuntimeError, match="\\[B,4\\]"):
        model.project_global(tokens)


def test_query_and_field_perturbations_change_outputs_and_diagnostics_are_finite() -> None:
    module = _module()
    torch.manual_seed(11)
    model = module.RAELActionCategoryFoundation()
    reader = _FormalP3Reader()
    baseline = model(reader, _prepared(batch=1))

    with torch.no_grad():
        model.stop_query.add_(torch.linspace(0.1, -0.1, model.dim))
    query_changed = model(reader, _prepared(batch=1))
    field_changed = model(reader, _prepared(batch=1, perturb=0.37))

    assert not torch.allclose(baseline["action_visual_tokens"], query_changed["action_visual_tokens"])
    assert not torch.allclose(baseline["action_visual_tokens"], field_changed["action_visual_tokens"])
    assert baseline["z_A_global_visual"].shape == (1, 4)
    diagnostics = baseline["diagnostics"]
    assert diagnostics["action_token_norm"].shape == (1, 4)
    assert diagnostics["readout_norm"].shape == (1, 4)
    assert diagnostics["layer_weight_entropy"].shape == (1, 4)
    for value in (*baseline.values(), *diagnostics.values()):
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()


def test_real_p3_read_preserves_all_four_layers_without_p7_layer_mean_or_concat() -> None:
    module = _module()
    from fate_oia.models.rael_multilayer_field import RAELMultiLayerField

    torch.manual_seed(23)
    field = RAELMultiLayerField()
    field.eval()
    model = module.RAELActionCategoryFoundation()
    with torch.no_grad():
        prepared = field.precompute(
            torch.randn(1, 4, 3600, 384),
            torch.randn(1, 4, 384),
        )
    output = model(field, prepared)

    assert output["readouts"].shape == (1, 4, 384)
    assert output["layer_weights"].shape == (1, 4, 4)
    assert torch.allclose(output["layer_weights"].sum(dim=-1), torch.ones(1, 4), atol=1e-6)
    source = inspect.getsource(module.RAELActionCategoryFoundation)
    assert 'group_name="action_category"' in source
    assert "layer_mean" not in source
    assert ".mean(dim=1)" not in source


def test_rejects_wrong_p3_shapes_and_keeps_dtype_device_boundary_finite() -> None:
    module = _module()
    model = module.RAELActionCategoryFoundation()

    class _WrongReader:
        def read(self, prepared: dict[str, Tensor], queries: Tensor, group_name: str | None = None) -> dict[str, Tensor]:
            return {
                "readout": queries[:, :3],
                "layer_weights": torch.ones((queries.shape[0], 3, 4), device=queries.device),
            }

    with pytest.raises(ValueError, match="readout"):
        model(_WrongReader(), _prepared())

    output = model(_FormalP3Reader(), _prepared(batch=1))
    assert output["action_visual_tokens"].device == next(model.parameters()).device
    assert output["action_visual_tokens"].dtype == next(model.parameters()).dtype
    assert torch.isfinite(output["action_visual_tokens"]).all()

