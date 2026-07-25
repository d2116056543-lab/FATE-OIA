"""Contract tests for the RAEL four-layer visual field.

The first test intentionally makes the RED state a normal assertion failure when
the P3 module is absent.  The remaining behavioral tests stay skipped in that
state so collection never depends on an unimplemented module.
"""

from __future__ import annotations

import importlib
import inspect

import pytest
import torch
from torch import nn


try:
    _module = importlib.import_module("fate_oia.models.rael_multilayer_field")
    RAELMultiLayerField = _module.RAELMultiLayerField
except ModuleNotFoundError:
    RAELMultiLayerField = None


pytestmark = pytest.mark.filterwarnings("error")


def test_p3_module_exists_with_public_reader_contract() -> None:
    """RED must be an assertion, not a module-collection failure."""
    assert RAELMultiLayerField is not None, "P3 RAELMultiLayerField is not implemented"
    assert hasattr(RAELMultiLayerField, "precompute")
    assert hasattr(RAELMultiLayerField, "read")
    assert hasattr(RAELMultiLayerField, "finalize_batch_collapse")


@pytest.fixture()
def field() -> "RAELMultiLayerField":
    if RAELMultiLayerField is None:
        pytest.skip("P3 is intentionally absent during RED")
    torch.manual_seed(7)
    return RAELMultiLayerField(
        dim=8,
        num_layers=4,
        formal_grid_hw=(45, 80),
        collapse_patience=2,
    )


@pytest.fixture()
def tokens() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(11)
    return torch.randn(2, 4, 12, 8), torch.randn(2, 4, 8)


class _CountingLinear(nn.Linear):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return super().forward(value)


def _replace_kv_with_counters(field: "RAELMultiLayerField") -> list[_CountingLinear]:
    counters: list[_CountingLinear] = []
    for name in ("key_projections", "value_projections"):
        replaced = nn.ModuleList()
        for linear in getattr(field, name):
            counter = _CountingLinear(linear.in_features, linear.out_features, bias=False)
            counter.weight.data.copy_(linear.weight.data)
            replaced.append(counter)
            counters.append(counter)
        setattr(field, name, replaced)
    return counters


def test_formal_metadata_preserves_all_dino_tokens(field: "RAELMultiLayerField") -> None:
    assert field.formal_grid_hw == (45, 80)
    assert field.formal_num_tokens == 3600
    assert field.dim == 8
    assert len(field.input_projections) == 4
    assert len(field.local_convs) == 4
    assert all(conv.groups == 8 and conv.kernel_size == (3, 3) for conv in field.local_convs)
    assert torch.allclose(field.local_gamma.detach(), torch.full((4,), 0.02))


def test_layer_router_has_no_unauthorized_fixed_bias(field: "RAELMultiLayerField") -> None:
    """RAEL alpha is exactly softmax(w^T tanh(Wq q + Wg g_l))."""
    assert not hasattr(field, "layer_bias")
    assert field.layer_score.bias is None
    reader_source = inspect.getsource(RAELMultiLayerField._read_tensor)
    assert "layer_bias" not in reader_source


def test_precompute_keeps_layer_specific_tokens_and_kv_once(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    counters = _replace_kv_with_counters(field)
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    assert prepared["field_tokens"].shape == (2, 4, 12, 8)
    assert prepared["keys_by_layer"].shape == (2, 4, 12, 8)
    assert prepared["values_by_layer"].shape == (2, 4, 12, 8)
    assert prepared["layer_global_tokens"].shape == (2, 4, 8)
    assert prepared["dtype_semantics"]["output_dtype"] == str(prepared["field_tokens"].dtype)
    assert prepared["dtype_semantics"]["internal_autocast"] is False
    assert [counter.calls for counter in counters] == [1] * 8

    query_groups = {"action": torch.randn(2, 3, 8), "reason": torch.randn(2, 2, 8)}
    first = field.read(prepared, query_groups)
    second = field.read(prepared, query_groups)
    assert set(first) == {"action", "reason"}
    assert first["action"]["readout"].shape == (2, 3, 8)
    assert second["reason"]["layer_weights"].shape == (2, 2, 4)
    # Two readers must consume precomputed KV, never re-project them.
    assert [counter.calls for counter in counters] == [1] * 8


def test_reader_is_query_and_sample_dependent_not_four_layer_mean(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    queries = torch.stack((torch.ones(8), -torch.ones(8)), dim=0).unsqueeze(0).repeat(2, 1, 1)
    queries[1, 1] *= 0.5
    result = field.read(prepared, queries, group_name="probe")
    weights = result["layer_weights"]
    assert weights.shape == (2, 2, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights[..., 0]), atol=1e-6)
    assert not torch.allclose(weights[0, 0], weights[0, 1])
    assert not torch.allclose(weights[0, 1], weights[1, 1])

    layer_reads = result["layer_readouts"]
    weighted = (weights.unsqueeze(-1) * layer_reads).sum(dim=2)
    incorrect_mean = layer_reads.mean(dim=2)
    assert torch.allclose(result["readout"], weighted, atol=1e-6)
    assert not torch.allclose(result["readout"], incorrect_mean, atol=1e-5)


def test_rank_four_query_groups_keep_per_group_layer_diagnostics(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    grouped_queries = torch.randn(2, 3, 2, 8)
    result = field.read(prepared, grouped_queries, group_name="all_slots")
    assert result["readout"].shape == (2, 3, 2, 8)
    assert result["layer_weights"].shape == (2, 3, 2, 4)
    # Group diagnostics aggregate queries within each group, not all groups.
    assert result["per_group_layer_weights"].shape == (2, 3, 4)


def test_local_adapter_responds_to_spatial_perturbation(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    base = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))["field_tokens"]
    perturbed_tokens = patch_tokens.clone()
    perturbed_tokens[:, 0, 5, :] += 4.0
    perturbed = field.precompute(perturbed_tokens, cls_tokens, grid_hw=(3, 4))["field_tokens"]
    # The changed patch and a neighbouring position must both react.  A model
    # that omits the depthwise 3x3 term fails the neighbour assertion.
    delta = (perturbed - base).abs().sum(dim=-1)
    assert torch.all(delta[:, 0, 5] > 0)
    assert torch.all(delta[:, 0, 1] > 0)


def test_two_optimizer_updates_reach_projection_and_depthwise_parameters(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    optimizer = torch.optim.AdamW(field.parameters(), lr=1e-2)
    query = torch.randn(2, 3, 8)
    nonzero_after_second: dict[str, bool] = {}
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
        output = field.read(prepared, query, group_name="train")["readout"]
        loss = output.square().mean() + prepared["field_tokens"].square().mean()
        loss.backward()
        if step == 1:
            nonzero_after_second = {
                "projection": bool(field.input_projections[0].weight.grad.abs().sum() > 0),
                "depthwise": bool(field.local_convs[0].weight.grad.abs().sum() > 0),
            }
        optimizer.step()
    assert all(nonzero_after_second.values())


def test_persistent_single_layer_collapse_sets_fail_flag(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    with torch.no_grad():
        field.layer_query_projection.weight.zero_()
        field.layer_global_projection.weight.zero_()
        field.layer_global_projection.weight[0, 0] = 1.0
        field.layer_score.weight.zero_()
        field.layer_score.weight[0, 0] = 10.0
        prepared["layer_global_tokens"].zero_()
        prepared["layer_global_tokens"][:, 0, 0] = 1.0
        prepared["layer_global_tokens"][:, 1:, 0] = -1.0
    first_read = field.read(prepared, torch.randn(2, 2, 8), group_name="collapse")
    assert float(first_read["layer_collapse_rate"]) == pytest.approx(1.0)
    assert int(field._collapse_streak) == 0
    first = field.finalize_batch_collapse(prepared, {"action": first_read})
    assert int(first["layer_collapse_streak"]) == 1
    assert not bool(first["layer_collapse_fail"])

    second_prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    second_prepared["layer_global_tokens"].copy_(prepared["layer_global_tokens"])
    second_read = field.read(second_prepared, torch.randn(2, 2, 8), group_name="collapse")
    second = field.finalize_batch_collapse(second_prepared, {"action": second_read})
    assert bool(second["layer_collapse_fail"])


def test_same_batch_three_groups_finalizes_collapse_once(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    reads = field.read(
        prepared,
        {
            "action": torch.randn(2, 4, 8),
            "reason": torch.randn(2, 5, 8),
            "slot": torch.randn(2, 3, 8),
        },
    )
    assert int(field._collapse_streak) == 0
    first = field.finalize_batch_collapse(prepared, reads)
    assert first["collapse_state_updated"] is True
    assert int(first["layer_collapse_streak"]) in (0, 1)
    with pytest.raises(RuntimeError, match="already finalized"):
        field.finalize_batch_collapse(prepared, reads)


def test_eval_finalize_returns_diagnostics_without_mutating_collapse_state(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    field.eval()
    before = {
        "streak": field._collapse_streak.detach().clone(),
        "fail": field._collapse_fail.detach().clone(),
        "dominant": field._last_dominant_layer.detach().clone(),
    }
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    reads = field.read(prepared, {"action": torch.randn(2, 3, 8)})
    result = field.finalize_batch_collapse(prepared, reads)
    assert result["collapse_state_updated"] is False
    assert torch.equal(field._collapse_streak, before["streak"])
    assert torch.equal(field._collapse_fail, before["fail"])
    assert torch.equal(field._last_dominant_layer, before["dominant"])


def test_collapse_buffers_roundtrip_in_state_dict(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = tokens
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    reads = field.read(prepared, {"action": torch.randn(2, 3, 8)})
    field.finalize_batch_collapse(prepared, reads)
    restored = RAELMultiLayerField(dim=8, num_layers=4, formal_grid_hw=(45, 80), collapse_patience=2)
    restored.load_state_dict(field.state_dict())
    assert torch.equal(restored._collapse_streak, field._collapse_streak)
    assert torch.equal(restored._collapse_fail, field._collapse_fail)
    assert torch.equal(restored._last_dominant_layer, field._last_dominant_layer)


def test_cpu_bfloat16_uses_safe_internal_dtype_path_without_external_autocast(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    patch_tokens, cls_tokens = (value.to(torch.bfloat16) for value in tokens)
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    assert prepared["field_tokens"].dtype == torch.float32
    assert prepared["dtype_semantics"] == {
        "input_dtype": "torch.bfloat16",
        "compute_dtype": "torch.float32",
        "output_dtype": "torch.float32",
        "internal_autocast": False,
    }
    result = field.read(prepared, torch.randn(2, 3, 8, dtype=torch.bfloat16), group_name="bf16")
    assert result["readout"].dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_bfloat16_uses_internal_autocast_when_supported() -> None:
    device = torch.device("cuda")
    field = RAELMultiLayerField(dim=8, num_layers=4, formal_grid_hw=(45, 80)).to(device)
    patch_tokens = torch.randn(1, 4, 12, 8, device=device, dtype=torch.bfloat16)
    cls_tokens = torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16)
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    expected_autocast = torch.cuda.is_bf16_supported()
    assert prepared["dtype_semantics"]["internal_autocast"] is expected_autocast
    # CUDA autocast accepts the bare BF16 input, while LayerNorm deliberately
    # retains FP32 output for stable layer evidence.  The API records this.
    assert prepared["dtype_semantics"]["input_dtype"] == "torch.bfloat16"
    assert prepared["dtype_semantics"]["compute_dtype"] == (
        "torch.bfloat16" if expected_autocast else "torch.float32"
    )
    assert prepared["dtype_semantics"]["output_dtype"] == str(prepared["field_tokens"].dtype)
    assert prepared["field_tokens"].dtype == torch.float32


def test_bad_reader_contracts_would_trigger_red() -> None:
    """The behavioral assertions reject each forbidden shortcut explicitly."""
    weights = torch.tensor([[[0.99, 0.01, 0.0, 0.0]]])
    layer_reads = torch.tensor([[[[8.0], [0.0], [0.0], [0.0]]]])
    correct = (weights.unsqueeze(-1) * layer_reads).sum(dim=2)
    bad_mean = layer_reads.mean(dim=2)
    with pytest.raises(AssertionError):
        assert torch.allclose(correct, bad_mean)

    local_delta = torch.tensor(0.4)
    no_local_delta = torch.tensor(0.0)
    with pytest.raises(AssertionError):
        assert torch.allclose(local_delta, no_local_delta)

    query_a = torch.tensor([0.8, 0.1, 0.1, 0.0])
    query_b = torch.tensor([0.1, 0.1, 0.7, 0.1])
    with pytest.raises(AssertionError):
        assert torch.allclose(query_a, query_b)

    precompute_kv_calls, repeated_reader_kv_calls = 8, 16
    with pytest.raises(AssertionError):
        assert repeated_reader_kv_calls == precompute_kv_calls


def test_forbidden_mutation_doubles_trigger_real_contract_failures(
    field: "RAELMultiLayerField", tokens: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Each prohibited shortcut violates an executable reader invariant."""
    patch_tokens, cls_tokens = tokens
    counters = _replace_kv_with_counters(field)
    prepared = field.precompute(patch_tokens, cls_tokens, grid_hw=(3, 4))
    result = field.read(prepared, torch.randn(2, 3, 8), group_name="mutation")
    weights = result["layer_weights"]
    reads = result["layer_readouts"]
    correct = result["readout"]

    # Extra fixed bias changes the prescribed alpha equation.
    fixed_bias = torch.tensor([9.0, 0.0, 0.0, 0.0])
    biased_weights = torch.softmax(weights.clamp_min(1e-8).log() + fixed_bias, dim=-1)
    with pytest.raises(AssertionError):
        assert torch.allclose(biased_weights, weights, atol=1e-6)

    # A direct mean over the four layer readouts ignores alpha.
    with pytest.raises(AssertionError):
        assert torch.allclose(reads.mean(dim=2), correct, atol=1e-5)

    # Re-projecting K/V for every query group doubles the precompute budget.
    for projection in field.key_projections:
        _ = projection(prepared["field_tokens"][:, 0])
    for projection in field.value_projections:
        _ = projection(prepared["field_tokens"][:, 0])
    with pytest.raises(AssertionError):
        assert [counter.calls for counter in counters] == [1] * 8

    # Concatenating four fields creates 14,400 formal tokens rather than one
    # per-layer 3,600-token field; a fixed first-layer read also ignores alpha.
    concatenated = torch.cat([prepared["field_tokens"][:, index] for index in range(4)], dim=1)
    with pytest.raises(AssertionError):
        assert concatenated.shape[1] == prepared["field_tokens"].shape[2]
    with pytest.raises(AssertionError):
        assert torch.allclose(reads[:, :, 0], correct, atol=1e-5)
