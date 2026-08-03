import pytest
import torch
import torch.nn.functional as F

from fate_oia.losses.save_pu_losses import (
    BBAMPrototypeBank,
    balanced_angular_margin_loss,
    build_bbam_tail_spec,
    save_bbam_loss,
    select_tail_reason_ids,
)
from fate_oia.losses.save_loss_registry import SAVELossRegistry


def _targets_from_counts(counts: list[int]) -> torch.Tensor:
    rows = torch.zeros(max(counts), len(counts))
    for label, count in enumerate(counts):
        rows[:count, label] = 1.0
    return rows


def test_bbam_tail_ids_are_bottom_eight_from_train_main_only() -> None:
    targets = _targets_from_counts([10, 1, 9, 2, 8, 3, 7, 4, 6, 5])

    tail_ids = select_tail_reason_ids(targets, split_name="train_main", tail_count=8)
    spec = build_bbam_tail_spec(targets, split_name="train_main", tail_count=8)

    assert set(tail_ids) == {1, 3, 4, 5, 6, 7, 8, 9}
    assert spec["source_split"] == "train_main"
    assert spec["artifact"] == "artifacts/save/tail_reason_ids.json"
    assert spec["tail_reason_ids"] == tail_ids


def test_bbam_refuses_non_train_main_tail_provenance() -> None:
    targets = _targets_from_counts([3] * 21)

    with pytest.raises(ValueError, match="train_main"):
        select_tail_reason_ids(targets, split_name="test", tail_count=8)


def test_bbam_gradient_is_owned_by_tail_reason_labels_only() -> None:
    torch.manual_seed(7)
    embeddings = torch.randn(4, 21, 8, requires_grad=True)
    targets = torch.randint(0, 2, (4, 21)).float()
    positive = torch.randn(21, 8)
    negative = torch.randn(21, 8)
    tail_ids = list(range(8))

    loss = balanced_angular_margin_loss(
        embeddings,
        targets,
        tail_ids,
        positive_prototypes=positive,
        negative_prototypes=negative,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert float(embeddings.grad[:, :8].abs().sum()) > 0.0
    assert torch.equal(embeddings.grad[:, 8:], torch.zeros_like(embeddings.grad[:, 8:]))


def test_formal_bbam_fails_closed_without_distinct_prototypes() -> None:
    embeddings = torch.randn(2, 3, 4)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="prototype"):
        balanced_angular_margin_loss(embeddings, targets, [0])
    with pytest.raises(ValueError, match="prototype bank"):
        save_bbam_loss(embeddings, targets, tail_reason_ids=[0])

    same = torch.ones(3, 4)
    with pytest.raises(ValueError, match="distinct"):
        balanced_angular_margin_loss(
            embeddings,
            targets,
            [0],
            positive_prototypes=same,
            negative_prototypes=same.clone(),
        )


def test_formal_bbam_has_useful_nonzero_angular_gradient() -> None:
    embeddings = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negative = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])

    loss = balanced_angular_margin_loss(
        embeddings,
        targets,
        [0],
        positive_prototypes=positive,
        negative_prototypes=negative,
    )
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert float(embeddings.grad[0, 0, 0]) < 0.0
    assert torch.equal(embeddings.grad[:, 1], torch.zeros_like(embeddings.grad[:, 1]))


def test_bbam_bank_exact_ema_counters_tail_mutation_and_state_restore() -> None:
    bank = BBAMPrototypeBank(
        reason_dim=4,
        embedding_dim=2,
        tail_reason_ids=[1, 3],
        momentum=0.5,
    )
    assert torch.equal(bank.positive_prototypes, torch.zeros(4, 2))
    assert torch.equal(bank.negative_prototypes, torch.zeros(4, 2))
    assert torch.equal(bank.positive_updates, torch.zeros(4, dtype=torch.long))
    assert torch.equal(bank.negative_updates, torch.zeros(4, dtype=torch.long))

    targets = torch.tensor([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    first = torch.zeros(2, 4, 2)
    first[0, 1], first[1, 1] = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    first[0, 3], first[1, 3] = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    bank.update(first, targets, split_name="train_main")

    torch.testing.assert_close(bank.positive_prototypes[1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(bank.negative_prototypes[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(bank.positive_prototypes[3], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(bank.negative_prototypes[3], torch.tensor([1.0, 0.0]))
    assert torch.equal(bank.positive_updates, torch.tensor([0, 1, 0, 1]))
    assert torch.equal(bank.negative_updates, torch.tensor([0, 1, 0, 1]))

    second = torch.zeros(2, 4, 2)
    second[0, 1], second[1, 1] = torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])
    second[0, 3], second[1, 3] = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    bank.update(second, targets, split_name="train_main")
    diagonal = F.normalize(torch.tensor([0.5, 0.5]), dim=0)

    for label in (1, 3):
        torch.testing.assert_close(bank.positive_prototypes[label], diagonal)
        torch.testing.assert_close(bank.negative_prototypes[label], diagonal)
    assert torch.equal(bank.positive_prototypes[[0, 2]], torch.zeros(2, 2))
    assert torch.equal(bank.negative_prototypes[[0, 2]], torch.zeros(2, 2))
    assert torch.equal(bank.positive_updates, torch.tensor([0, 2, 0, 2]))
    assert torch.equal(bank.negative_updates, torch.tensor([0, 2, 0, 2]))

    restored = BBAMPrototypeBank(
        reason_dim=4,
        embedding_dim=2,
        tail_reason_ids=[1, 3],
        momentum=0.5,
    )
    restored.load_state_dict(bank.state_dict())
    for name, value in bank.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)


def test_save_bbam_is_raw_and_registry_applies_the_only_weight() -> None:
    bank = BBAMPrototypeBank(
        reason_dim=2,
        embedding_dim=2,
        tail_reason_ids=[0],
        momentum=0.5,
    )
    bank_embeddings = torch.tensor(
        [[[1.0, 0.0], [0.0, 0.0]], [[-1.0, 0.0], [0.0, 0.0]]]
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    bank.update(bank_embeddings, targets, split_name="train_main")
    embeddings = torch.tensor(
        [[[0.0, 1.0], [0.0, 0.0]], [[0.0, -1.0], [0.0, 0.0]]],
        requires_grad=True,
    )

    raw = balanced_angular_margin_loss(
        embeddings,
        targets,
        [0],
        positive_prototypes=bank.positive_prototypes,
        negative_prototypes=bank.negative_prototypes,
    )
    wrapped = save_bbam_loss(
        embeddings,
        targets,
        tail_reason_ids=[0],
        prototype_bank=bank,
    )
    registry = SAVELossRegistry(expected_terms=("reason_bbam",))
    registry.add("reason_bbam", wrapped)

    assert float(raw.detach()) > 0.0
    torch.testing.assert_close(wrapped, raw)
    torch.testing.assert_close(registry.total(), raw * 0.03)
