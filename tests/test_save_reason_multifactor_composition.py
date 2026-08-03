import copy

import pytest
import torch

try:
    from fate_oia.losses.save_reason_losses import (
        SAVE_REASON_LOSS_WEIGHTS,
        balanced_angular_margin_loss,
        save_reason_loss,
        select_tail_reason_ids,
    )
    from fate_oia.models.save_reason_decoder import SAVEPrivateReasonDecoder
except ImportError:
    SAVE_REASON_LOSS_WEIGHTS = None
    SAVEPrivateReasonDecoder = None
    balanced_angular_margin_loss = None
    save_reason_loss = None
    select_tail_reason_ids = None


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "reason_logits_clean": torch.randn(1, 3),
        "global_field": torch.randn(1, 17, 8),
        "detail_field": torch.randn(1, 17, 8),
        "factor_measurement_token": torch.randn(1, 3, 8),
        "factor_evidence_map": torch.rand(1, 3, 17),
        "factor_reliability": torch.ones(1, 3),
    }


def _zero_module(module: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.zero_()


@pytest.mark.parametrize(
    "module_name",
    ["global_cross_attention", "detail_cross_attention", "reason_self_attention"],
)
def test_private_reason_first_read_components_are_independently_active(
    module_name: str,
) -> None:
    torch.manual_seed(10)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    ablated = copy.deepcopy(decoder)
    _zero_module(getattr(ablated, module_name))
    inputs = _inputs()

    baseline = decoder(**inputs, progress=1.0)
    changed = ablated(**inputs, progress=1.0)

    assert not torch.allclose(
        baseline["reason_logits_private_direct"],
        changed["reason_logits_private_direct"],
    )


def test_evidence_map_composition_independently_controls_detail_reread() -> None:
    torch.manual_seed(13)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    inputs = _inputs()
    baseline = decoder(**inputs, progress=1.0)
    changed_inputs = {key: value.clone() for key, value in inputs.items()}
    changed_inputs["factor_evidence_map"] = torch.flip(
        changed_inputs["factor_evidence_map"],
        dims=(-1,),
    )
    changed = decoder(**changed_inputs, progress=1.0)

    assert not torch.allclose(
        baseline["reason_composed_evidence_map"],
        changed["reason_composed_evidence_map"],
    )
    assert not torch.allclose(
        baseline["reason_detail_reread_attention"],
        changed["reason_detail_reread_attention"],
    )


def test_detail_reread_is_independently_active() -> None:
    torch.manual_seed(14)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    ablated = copy.deepcopy(decoder)
    _zero_module(ablated.reread_output)
    inputs = _inputs()

    baseline = decoder(**inputs, progress=1.0)
    changed = ablated(**inputs, progress=1.0)

    assert not torch.allclose(
        baseline["reason_logits_private_direct"],
        changed["reason_logits_private_direct"],
    )


def test_private_reason_composes_multiple_factors_and_re_reads_evidence() -> None:
    if SAVEPrivateReasonDecoder is None:
        pytest.fail("SAVEPrivateReasonDecoder is not implemented")

    torch.manual_seed(11)
    decoder = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    inputs = _inputs()
    first = decoder(**inputs, progress=1.0)

    changed_first = {key: value.clone() for key, value in inputs.items()}
    changed_first["factor_measurement_token"][:, 0] += 2.0
    first_changed = decoder(**changed_first, progress=1.0)
    changed_second = {key: value.clone() for key, value in inputs.items()}
    changed_second["factor_measurement_token"][:, 1] -= 2.0
    second_changed = decoder(**changed_second, progress=1.0)

    factor_attention = first["reason_factor_attention"]
    assert factor_attention.shape == (1, 3, 4)
    assert torch.count_nonzero(factor_attention[0, 0, :3] > 1e-5) >= 2
    assert first["reason_composed_evidence_map"].shape == (1, 3, 17)
    assert first["reason_detail_reread_attention"].shape == (1, 3, 17)
    assert not torch.allclose(
        first["reason_logits_private_direct"],
        first_changed["reason_logits_private_direct"],
    )
    assert not torch.allclose(
        first["reason_logits_private_direct"],
        second_changed["reason_logits_private_direct"],
    )


def test_reason_loss_uses_exact_section_16_2_weights_and_train_main_tail_ids() -> None:
    if save_reason_loss is None:
        pytest.fail("SAVE reason loss is not implemented")

    assert SAVE_REASON_LOSS_WEIGHTS == {
        "benchmark": 1.00,
        "private_direct": 0.35,
        "clean": 0.35,
        "rank": 0.06,
        "soft_f1": 0.03,
        "bbam": 0.03,
        "view_consistency": 0.02,
        "pu_private": 1.00,
    }

    torch.manual_seed(12)
    output = {
        "reason_logits_benchmark": torch.randn(4, 5, requires_grad=True),
        "reason_logits_private_direct": torch.randn(4, 5, requires_grad=True),
        "reason_logits_clean": torch.randn(4, 5, requires_grad=True),
        "reason_reliability": torch.rand(4, 5),
        "reason_embedding_private": torch.randn(4, 5, 8, requires_grad=True),
        "reason_logits_pu_private": torch.randn(4, 5, requires_grad=True),
    }
    target = torch.randint(0, 2, (4, 5)).float()
    losses = save_reason_loss(output, target, tail_reason_ids=torch.tensor([1, 3]))
    expected = (
        losses["benchmark"]
        + 0.35 * losses["private_direct"]
        + 0.35 * losses["clean"]
        + 0.06 * losses["rank"]
        + 0.03 * losses["soft_f1"]
        + 0.03 * losses["bbam"]
        + 0.02 * losses["view_consistency"]
        + losses["pu_private"]
    )
    torch.testing.assert_close(losses["total"], expected)

    train_main = torch.tensor(
        [[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 1, 0, 0]],
        dtype=torch.float32,
    )
    tail_ids = select_tail_reason_ids(train_main, tail_count=2)
    assert tail_ids.tolist() == [1, 3]

    embedding = torch.randn(3, 5, 8)
    embedding[..., 0] = 0.0
    embedding.requires_grad_()
    positive_prototypes = torch.zeros(5, 8)
    negative_prototypes = torch.zeros(5, 8)
    positive_prototypes[:, 0] = 1.0
    negative_prototypes[:, 0] = -1.0
    bbam = balanced_angular_margin_loss(
        embedding,
        train_main,
        tail_reason_ids=tail_ids,
        positive_prototypes=positive_prototypes,
        negative_prototypes=negative_prototypes,
    )
    assert torch.isfinite(bbam)
    bbam.backward()
    assert embedding.grad is not None
    assert torch.count_nonzero(embedding.grad) > 0


def test_balanced_angular_margin_matches_signed_cosine_equation() -> None:
    embeddings = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [-1.0, 1.0], [0.0, 1.0]],
        ]
    )
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    positive = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    negative = torch.tensor([[0.0, -1.0], [-1.0, 0.0], [-1.0, -1.0]])
    tail_ids = torch.tensor([1, 2])
    margin = 0.15

    actual = balanced_angular_margin_loss(
        embeddings,
        target,
        tail_reason_ids=tail_ids,
        positive_prototypes=positive,
        negative_prototypes=negative,
        margin=margin,
    )
    selected = torch.nn.functional.normalize(embeddings[:, tail_ids], dim=-1)
    pos = torch.nn.functional.normalize(positive[tail_ids], dim=-1)
    neg = torch.nn.functional.normalize(negative[tail_ids], dim=-1)
    direction = 2.0 * target[:, tail_ids] - 1.0
    expected = torch.relu(
        margin
        - direction
        * ((selected * pos).sum(-1) - (selected * neg).sum(-1))
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_balanced_angular_margin_rewards_both_positive_and_negative_directions() -> None:
    target = torch.tensor([[1.0], [0.0]])
    positive = torch.tensor([[1.0, 0.0]])
    negative = torch.tensor([[-1.0, 0.0]])
    correct = torch.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]])
    reversed_direction = -correct

    correct_loss = balanced_angular_margin_loss(
        correct,
        target,
        tail_reason_ids=[0],
        positive_prototypes=positive,
        negative_prototypes=negative,
    )
    reversed_loss = balanced_angular_margin_loss(
        reversed_direction,
        target,
        tail_reason_ids=[0],
        positive_prototypes=positive,
        negative_prototypes=negative,
    )

    assert correct_loss == 0.0
    assert reversed_loss > correct_loss


def test_balanced_angular_margin_is_tail_only_and_non_tail_invariant() -> None:
    torch.manual_seed(15)
    embeddings = torch.randn(2, 3, 4, requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    positive = torch.randn(3, 4)
    negative = torch.randn(3, 4)
    baseline = balanced_angular_margin_loss(
        embeddings,
        target,
        tail_reason_ids=[1],
        positive_prototypes=positive,
        negative_prototypes=negative,
    )
    changed_embeddings = embeddings.detach().clone()
    changed_embeddings[:, [0, 2]] += 100.0
    changed_target = 1.0 - target
    changed_target[:, 1] = target[:, 1]
    changed = balanced_angular_margin_loss(
        changed_embeddings,
        changed_target,
        tail_reason_ids=[1],
        positive_prototypes=positive,
        negative_prototypes=negative,
    )

    torch.testing.assert_close(changed, baseline.detach())
    baseline.backward()
    assert torch.count_nonzero(embeddings.grad[:, 1]) > 0
    assert torch.count_nonzero(embeddings.grad[:, [0, 2]]) == 0
