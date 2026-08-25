import torch

from fate_oia.losses.tida_relational_traffic_losses import (
    relational_deletion_contrast_loss,
    relational_proper_no_harm_loss,
)


def test_selected_deletion_must_hurt_target_margin_more_than_random():
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    full = torch.tensor([[0.08, -0.08], [-0.08, 0.08]], requires_grad=True)
    selected_deleted = torch.zeros_like(full)
    random_deleted = 0.75 * full.detach()
    support = torch.ones_like(full)
    good = relational_deletion_contrast_loss(
        full, selected_deleted, random_deleted, target, support, margin=0.01
    )
    bad = relational_deletion_contrast_loss(
        full, random_deleted, selected_deleted, target, support, margin=0.01
    )
    assert good < bad
    assert bad > 1.0
    bad.backward()
    assert torch.isfinite(full.grad).all()


def test_deletion_contrast_respects_pu_element_weights():
    target = torch.tensor([[1.0, 0.0]])
    full = torch.tensor([[0.04, 0.04]])
    selected_deleted = torch.zeros_like(full)
    random_deleted = full.clone()
    support = torch.ones_like(full)
    positive_only = relational_deletion_contrast_loss(
        full, selected_deleted, random_deleted, target, support,
        element_weight=torch.tensor([[1.0, 0.0]]),
    )
    all_labels = relational_deletion_contrast_loss(
        full, selected_deleted, random_deleted, target, support,
        element_weight=torch.ones_like(full),
    )
    assert positive_only < all_labels


def test_relational_proper_no_harm_only_penalizes_worse_delta():
    base = torch.tensor([[0.2, -0.2]])
    target = torch.tensor([[1.0, 0.0]])
    better = torch.tensor([[0.3, -0.3]], requires_grad=True)
    worse = torch.tensor([[-0.3, 0.3]], requires_grad=True)

    better_loss = relational_proper_no_harm_loss(base, better, target)
    worse_loss = relational_proper_no_harm_loss(base, worse, target)

    assert better_loss.item() == 0.0
    assert worse_loss.item() > 1.0
    worse_loss.backward()
    assert worse.grad is not None and torch.isfinite(worse.grad).all()
